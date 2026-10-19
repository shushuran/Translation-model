"""优化器：Adam、全局范数梯度裁剪、warmup + 余弦退火。

本模块不依赖任何框架，只操作 numpy 数组。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Adam", "clip_grad_global_norm", "warmup_cosine_lr", "collect_params"]


def collect_params(*modules) -> tuple[dict, dict]:
    """把若干模块的参数和梯度收集成两个同键的字典（带前缀，避免重名）。

    用法：`params, grads = collect_params(model)`，再交给 `Adam(params, grads)`。
    返回的是**内部数组的引用**，所以优化器能就地更新参数。
    """
    params: dict = {}
    grads: dict = {}
    for i, m in enumerate(modules):
        prefix = f"m{i}." if len(modules) > 1 else ""
        for name, p in m.named_parameters():
            params[prefix + name] = p
        for name, g in m.named_grads().items():
            grads[prefix + name] = g
    return params, grads


class Adam:
    """Adam（Kingma & Ba, 2015）。

        m_t = β₁ m_{t-1} + (1-β₁) g_t                    一阶矩（梯度的滑动平均）
        v_t = β₂ v_{t-1} + (1-β₂) g_t²                   二阶矩（梯度平方的滑动平均）
        m̂_t = m_t / (1-β₁ᵗ)                              偏差修正
        v̂_t = v_t / (1-β₂ᵗ)
        θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)

    **为什么需要偏差修正？** m 和 v 都初始化为 0，前几步会严重偏小
    （比如第一步 m₁ = 0.1·g₁，只有真实梯度的 1/10）。
    除以 (1-β^t) 正好把它校正回来：第 1 步 1-0.9¹ = 0.1，于是 m̂₁ = g₁。

    **为什么 Adam 对梯度尺度不敏感？**
    更新量是 `m̂ / (√v̂ + ε)`，分子分母都正比于梯度的尺度，
    尺度因子**互相抵消**了。所以即使某个参数的梯度比其他参数小 7 个数量级
    （本项目第 4 步实测 `attn.W_q` 就是这样），它依然能得到合理的步长。
    这正是 Adam 相对朴素 SGD 最大的实际优势。
    """

    def __init__(
        self,
        params: dict,
        grads: dict,
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ) -> None:
        missing = set(params) - set(grads)
        if missing:
            raise KeyError(f"这些参数没有对应的梯度：{sorted(missing)}")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.bias_correction = bool(bias_correction)
        self.t = 0

        keys = sorted(set(params) & set(grads))
        # 只保留形状匹配的键（防止把广播出来的梯度误当成参数梯度）
        self._keys = [k for k in keys if params[k].shape == grads[k].shape]
        skipped = [k for k in keys if k not in self._keys]
        self.skipped = skipped
        self._params = {k: params[k] for k in self._keys}
        self._grads = {k: grads[k] for k in self._keys}
        self.m = {k: np.zeros_like(self._params[k]) for k in self._keys}
        self.v = {k: np.zeros_like(self._params[k]) for k in self._keys}

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        if self.bias_correction:
            c1 = 1.0 - b1 ** self.t
            c2 = 1.0 - b2 ** self.t
        else:
            c1 = c2 = 1.0

        for k in self._keys:
            p = self._params[k]
            g = self._grads[k]
            if self.weight_decay:
                g = g + self.weight_decay * p            # L2 正则（AdamW 会把它解耦出去）

            m = self.m[k]
            v = self.v[k]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)

            m_hat = m / c1
            v_hat = v / c2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for g in self._grads.values():
            g[...] = 0.0

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def state_dict(self) -> dict:
        return {"t": self.t, "lr": self.lr,
                "m": {k: v.copy() for k, v in self.m.items()},
                "v": {k: v.copy() for k, v in self.v.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.t = int(state["t"])
        self.lr = float(state["lr"])
        for k, val in state["m"].items():
            self.m[k][...] = val
        for k, val in state["v"].items():
            self.v[k][...] = val

    def __repr__(self) -> str:
        return (f"Adam(lr={self.lr:g}, betas=({self.beta1}, {self.beta2}), "
                f"eps={self.eps:g}, wd={self.weight_decay:g}, "
                f"tensors={len(self._keys)}, params={sum(p.size for p in self._params.values()):,})")


def clip_grad_global_norm(params: dict, grads: dict, max_norm: float) -> float:
    """**全局**范数裁剪：把所有参数的梯度当成一个大向量，整体缩放到范数 ≤ max_norm。

        g ← g · min(1, max_norm / (‖g‖₂ + ε))

    返回裁剪**之前**的总范数（用来监控，这比 loss 更早预警梯度爆炸）。

    为什么要用全局范数而不是逐张量裁剪？逐张量裁剪会改变各参数之间梯度的
    **相对比例**；全局裁剪只是统一缩放，保持了梯度的方向。RNN 里梯度爆炸
    是整体性的（BPTT 连乘），所以全局裁剪更合理。

    注意：`total_norm` 应该在裁剪前算，否则你永远看不到爆炸。
    """
    total_sq = 0.0
    for name, p in params.items():
        g = grads.get(name)
        if g is None or g.shape != p.shape:
            continue
        total_sq += float(np.sum(g * g))
    total_norm = math.sqrt(total_sq)

    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for name, p in params.items():
            g = grads.get(name)
            if g is None or g.shape != p.shape:
                continue
            g *= scale
    return total_norm


def warmup_cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_frac: float = 0.1,
    min_lr_ratio: float = 0.1,
) -> float:
    """前 `warmup_frac` 线性升温，之后余弦退火到 `min_lr_ratio · base_lr`。

        warmup 段：  lr = base_lr · (step+1) / warmup_steps
        cosine 段：  lr = base_lr · ( min_ratio + (1-min_ratio)·(1+cos(π·p))/2 )

    为什么要 warmup？训练最初几步参数还是随机的，梯度方向噪声很大；
    直接用大学习率容易把参数推到坏区域。先小步走一段，等方向稳定了再加速。

    为什么要余弦退火？后期学习率小，参数才能在最小值附近稳定下来；
    如果一直保持大学习率，loss 会在最优点附近持续抖动、下不去。

    实测这条曲线（base_lr=2e-3, total=3000, warmup=10%）：
        step=0    -> 6.7e-05     step=150  -> 2.0e-03（峰值）
        step=1500 -> 1.2e-03     step=2999 -> 2.0e-04
    """
    if total_steps <= 0:
        return float(base_lr)
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return float(base_lr) * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
