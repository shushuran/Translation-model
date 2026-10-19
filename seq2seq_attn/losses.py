"""交叉熵损失（带 padding mask），前向 + 反向。

    loss = - (1 / N) · Σ_{有效位置} log softmax(logits)[target]

其中 N 是**有效位置数**（mask=1 的个数），不是 batch 的总元素数。
这样不同 batch 里 padding 比例不同时，loss 的量级才可比、梯度尺度才稳定。

反向只有一行（这是交叉熵最优雅的地方）
------------------------------------
    ∂loss/∂logits = ( softmax(logits) - onehot(target) ) / N

推导：设 `p = softmax(z)`，`L = -log p_y`，则

    ∂L/∂z_j = -(1/p_y) · ∂p_y/∂z_j
            = -(1/p_y) · p_y (δ_yj - p_j)
            = p_j - δ_yj

`softmax` 的 Jacobian 和 `log` 的导数**互相抵消**了，结果就是"预测概率减真实标签"。
这就是为什么深度学习框架都提供融合的 `cross_entropy` 算子：
把 softmax 和 log 合在一起算，既省一次矩阵运算，又避免数值下溢。
"""

from __future__ import annotations

import numpy as np

from .nn import Module, log_softmax

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(Module):
    """softmax 交叉熵。没有可训练参数，但遵循 Module 的 forward/backward 协议。

    形状约定（支持任意前置维度，内部展平处理）
    ----------------------------------------
        logits  : (..., V)      V = 词表大小
        targets : (...)         int64 目标 id
        mask    : (...)         1.0 = 计入 loss，0.0 = 忽略（`<pad>` 位置）
                                不给 mask 时全部计入
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict | None = None
        # 诊断统计（不参与梯度）
        self.last_loss: float = float("nan")
        self.last_accuracy: float = float("nan")

    # ------------------------------------------------------------------ forward
    def forward(self, logits, targets, mask=None):
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        if logits.ndim < 2:
            raise ValueError(f"logits 至少要是二维 (..., V)，收到 {logits.shape}")
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"logits 的前置维度 {logits.shape[:-1]} 必须与 targets {targets.shape} 一致"
            )

        V = logits.shape[-1]
        shape = logits.shape
        n = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        flat_logits = logits.reshape(n, V)
        flat_targets = targets.reshape(-1)

        if flat_targets.size and (flat_targets.min() < 0 or flat_targets.max() >= V):
            raise IndexError(
                f"target id 超出词表范围 [0, {V})："
                f"min={int(flat_targets.min())}, max={int(flat_targets.max())}"
            )

        # 用 log_softmax 而不是 log(softmax(...))：后者会下溢成 -inf
        logp = log_softmax(flat_logits, axis=-1)
        picked = logp[np.arange(n), flat_targets]          # 目标词的 log 概率

        # 概率 = exp(log 概率)，与直接算 softmax 逐位一致，且这里已经算过了
        probs = np.exp(logp)

        if mask is None:
            weights = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
            n_valid = float(n)
        else:
            flat_mask = np.asarray(mask, dtype=np.float64).reshape(-1)
            if flat_mask.shape != flat_targets.shape:
                raise ValueError(
                    f"mask 形状 {flat_mask.shape} 与 targets {flat_targets.shape} 不一致"
                )
            n_valid = float(flat_mask.sum())
            weights = flat_mask / max(n_valid, 1.0)

        loss = float(-(picked * weights).sum())

        # 诊断统计
        correct = (flat_logits.argmax(-1) == flat_targets).astype(np.float64)
        if mask is None:
            acc = float(correct.mean()) if n else float("nan")
        else:
            acc = float((correct * weights).sum())

        self._cache = {
            "shape": shape,
            "targets": flat_targets,
            "probs": probs,
            "weights": weights,
        }
        self.last_loss = loss
        self.last_accuracy = acc
        return loss

    # ----------------------------------------------------------------- backward
    def backward(self, dloss: float = 1.0):
        """返回 dlogits，形状与 forward 的 logits 相同。

            dlogits = ( softmax(logits) - onehot(target) ) ⊙ weights · dloss
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        shape = c["shape"]
        n = c["probs"].shape[0]

        dlogits = c["probs"].copy()
        dlogits[np.arange(n), c["targets"]] -= 1.0
        dlogits *= c["weights"][:, None]
        dlogits *= float(dloss)
        return dlogits.reshape(shape)

    # ----------------------------------------------------------------- 无参数
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        return None
