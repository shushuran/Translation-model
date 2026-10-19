"""基础工具：参数容器、数值稳定的 softmax、数值梯度检验。

第 2 步只用到这里的一小部分；第 3 步的 Linear / Embedding / GRU 会继续往 nn.py 里加。

**为什么全程用 float64？**
数值梯度检验用的是中心差分 `(f(x+ε) - f(x-ε)) / 2ε`，ε 取 1e-6 时，
float32 只有 ~7 位有效数字，相减会产生灾难性抵消，相对误差会大到 1e-2，
根本没法判断解析梯度对不对。float64 有 ~16 位有效数字，相对误差能压到 1e-9。
本项目模型极小（几万参数、十几步序列），float64 的性能代价可以忽略。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "NEG_INF",
    "Module",
    "Linear",
    "Embedding",
    "init_uniform",
    "sigmoid",
    "log_softmax",
    "softmax",
    "softmax_masked",
    "softmax_backward",
    "numerical_gradient_at",
    "relative_error",
    "check_gradient",
    "summarize_gradient_report",
]

# 往 scores 上加的"极大负数"，即方案 A（加性 mask）。
# 注意力模块实际采用方案 B（-inf），保留这个常量是为了做两种方案的对照实验。
# 两者在正常量级下按位等价，差别在鲁棒性：详见 softmax_masked 的文档。
NEG_INF = -1e9


# --------------------------------------------------------------------------------------
# 参数初始化
# --------------------------------------------------------------------------------------
def init_uniform(shape: Sequence[int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """均匀初始化 U(-scale, scale)。

    千万不要全零初始化：那样同一层里所有神经元完全相同，
    反向传播拿到的梯度也完全相同，永远学不出差异（对称性无法打破）。
    """
    return rng.uniform(-scale, scale, size=tuple(shape)).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。

    朴素写法 `1/(1+exp(-x))` 在 x 很负时 `exp(-x)` 会溢出成 inf，得到 nan。
    这里对正负两支分别处理：
        x >= 0 : 1 / (1 + exp(-x))          —— exp(-x) <= 1，安全
        x <  0 : exp(x) / (1 + exp(x))      —— exp(x)  < 1，安全
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# --------------------------------------------------------------------------------------
# 参数容器
# --------------------------------------------------------------------------------------
class Module:
    """极简参数容器。登记参数、收集子模块、按名字取/写梯度、清零。

    设计上刻意让 `params()` 返回**内部字典的引用**而不是副本：
    数值梯度检验需要就地扰动参数数组，返回副本会导致扰动影响不到 forward。
    """

    def __init__(self) -> None:
        self._params: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._grads: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._modules: "OrderedDict[str, Module]" = OrderedDict()
        # 前向传播的中间量，backward 要用。
        # ⚠ 只有一个槽位 —— 见 save_cache 的说明。
        self._cache: dict | None = None

    # ---- 中间量的保存与恢复 ----
    def save_cache(self) -> dict | None:
        """把当前前向缓存快照下来（浅拷贝）。

        ⚠ 为什么需要这个：`_cache` 只有一个槽位，每次 `forward` 都会覆盖它。
        对普通前馈层没问题（前向一次、反向一次）。但**循环网络会在一个循环里
        调用同一个单元 S 次**，等反向传播时要逐个时间步回放 ——
        那时候 `_cache` 里只剩最后一次调用的中间量了，前 S-1 步全用错数据算梯度。

        这类 bug 非常隐蔽：形状全都对得上、不报错、不出 nan，
        只是梯度系统性错误（本项目实测相对误差直接到 1.0）。
        唯一的发现手段就是端到端的梯度检验。

        正确做法：前向循环里每步 `caches.append(cell.save_cache())`，
        反向循环里逐步 `cell.load_cache(caches[t])` 再 backward。
        这就是"用时间换空间"的经典取舍，也是 PyTorch autograd 保存
        `saved_tensors` 做的事。
        """
        return None if getattr(self, "_cache", None) is None else dict(self._cache)

    def load_cache(self, snapshot: dict | None) -> None:
        """恢复之前 `save_cache()` 得到的快照。"""
        self._cache = snapshot

    # ---- 登记参数 ----
    def _register(self, name: str, value: np.ndarray) -> np.ndarray:
        """登记一个可训练参数，并自动开一块同形状的梯度缓冲区（初始为 0）。"""
        arr = np.ascontiguousarray(value, dtype=np.float64)
        if name in self._params:
            raise KeyError(f"参数名重复：{name}")
        self._params[name] = arr
        self._grads[name] = np.zeros_like(arr)
        return arr

    # ---- 登记子模块 ----
    def register_module(self, name: str, module: "Module") -> "Module":
        """登记子模块，并返回它，方便写成 `self.cell = self.register_module("cell", GRUCell(...))`。

        为什么要显式登记：`named_parameters()` / `zero_grad()` / `num_params()`
        需要递归地走到子模块里去。不做登记的话，编码器自己的 `_params` 是空的，
        参数全在 `self.cell` 里，优化器就一个都拿不到。
        """
        if not isinstance(module, Module):
            raise TypeError(f"{name} 不是 Module 实例")
        self._modules[name] = module
        return module

    # ---- 访问 ----
    def params(self) -> "OrderedDict[str, np.ndarray]":
        """本层自己的参数（不含子模块）。"""
        return self._params

    def grads(self) -> "OrderedDict[str, np.ndarray]":
        """本层自己的梯度（不含子模块）。"""
        return self._grads

    def modules(self):
        """递归遍历自己 + 所有子模块。"""
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_parameters(self, prefix: str = ""):
        """递归产出 (带点号的参数名, 参数数组)。

        名字带前缀（如 `cell.W_hh`）是为了让优化器和梯度检验能看到参数属于谁。
        产出的是**内部数组的引用**，可以直接就地修改。
        """
        for name, p in self._params.items():
            yield prefix + name, p
        for mname, m in self._modules.items():
            yield from m.named_parameters(f"{prefix}{mname}.")

    def named_parameters_dict(self) -> dict:
        return dict(self.named_parameters())

    def named_grads(self, prefix: str = "") -> dict:
        out: dict = {}
        for name, g in self._grads.items():
            out[prefix + name] = g
        for mname, m in self._modules.items():
            out.update(m.named_grads(f"{prefix}{mname}."))
        return out

    def num_params(self) -> int:
        total = sum(p.size for p in self._params.values())
        return int(total + sum(m.num_params() for m in self._modules.values()))

    def zero_grad(self) -> None:
        for g in self._grads.values():
            g[...] = 0.0
        for m in self._modules.values():
            m.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        own = ", ".join(f"{k}{tuple(v.shape)}" for k, v in self._params.items())
        subs = ", ".join(f"{k}={v!r}" for k, v in self._modules.items())
        parts = [p for p in (own, subs) if p]
        return f"{type(self).__name__}(params={self.num_params()}" + (f": {'; '.join(parts)}" if parts else "") + ")"

    # ---- 存取权重 ----
    def state_dict(self) -> dict:
        """导出全部参数的**副本**（递归）。可直接交给 np.savez 存档。"""
        return {k: v.copy() for k, v in self.named_parameters()}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """把外部权重装载回去（**就地写入**，不替换数组对象）。

        为什么要就地写而不是直接换引用：优化器在构造时缓存了参数数组的引用，
        如果这里换成新数组，优化器就会继续更新旧的那一份，训练静默失效。
        """
        own = self.named_parameters_dict()
        if strict:
            missing = set(own) - set(state)
            extra = set(state) - set(own)
            if missing or extra:
                raise KeyError(f"权重键不匹配。缺失 {sorted(missing)}，多余 {sorted(extra)}")
        for name, param in own.items():
            if name not in state:
                continue
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != param.shape:
                raise ValueError(f"{name} 形状不符：期望 {param.shape}，收到 {value.shape}")
            param[...] = value


# --------------------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """标准 softmax，先减去最大值防止 exp 溢出。

    softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

    减最大值是**恒等变换**（分子分母同乘 exp(-max)），结果完全不变，
    但把 exp 的参数压到 <= 0，彻底避免 exp(1000) = inf。
    """
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_masked(scores: np.ndarray, mask: np.ndarray | None = None, axis: int = -1) -> np.ndarray:
    """带有效性 mask 的 softmax。

    参数
    ----
    scores : (..., S)
    mask   : 标记每个 key 是否有效，**1.0 表示有效、0.0 表示填充**。
             这正是 `Batch.src_mask` 的约定，可以直接传进来。
             允许三种形状，函数内部会统一处理：
                (S,)        —— 所有 batch、所有 query 共用同一个 mask。
                               右对齐后正好落在 key 维，语义无歧义。
                (B, S)      —— 最常用。mask 只描述「源句哪些位置是 padding」，
                               与 query 无关，所以不需要 Tq 这一维。
                (B, Tq, S)  —— 每个 query 有自己的 mask 时使用。

             注意 (B, S) 这一支必须显式插轴，不能靠自动广播 —— 见下文陷阱。
             格式不符（比如 scores 是 (B,Tq,S) 却传了四维 mask）会由 numpy 抛错。

    ⚠ numpy 广播的陷阱（本项目真的踩到过）
    ------------------------------------
    numpy 的广播规则是**从最右边一维开始向左对齐**。所以
        scores (B, Tq, S)  与  mask (B, S)
    会被对齐成
        (B, Tq, S)  与  (1, B, S)
    而不是我们想要的 `(B, 1, S)`。后果分两种情况：

        · B ≠ Tq 且两者都 > 1  → 直接抛 ValueError（还算幸运）
        · Tq == 1 或 B == 1    → **静默**广播成 (B, B, S)，
          形状看起来还挺合理，但每个 query 用的是**别的样本的 mask**，
          结果全错而且不报错。

    第二种情况恰好就是解码推理时的真实配置（Tq=1，B=32），
    所以这里必须显式 `expand_dims(mask, -2)` 把它变成 (B, 1, S)，不能依赖自动广播。

    两种实现方案
    -----------
    方案 A（加性 mask，`Batch.attn_mask()` 就是这么做的）：
        scores + (1 - mask) * (-1e9)，然后普通 softmax。

    方案 B（本函数采用）：
        np.where(mask, scores, -inf)，然后带保护的 softmax。

    **两个方案在正常量级下结果逐位完全相同**。这点值得澄清，因为网上常说方案 A
    "有精度损失"，实测并不成立：有效位置加的是精确的 0.0；填充位置
    `exp(-1e9 - max)` 在 float64 下下溢为精确的 0.0。两者的权重数组按位相等
    （可运行 scripts/02_attention.py 第 [3] 节复现）。

    真正的区别只有一处，但很重要 —— **某一行全部被 mask 掉时的语义**：
        方案 A 会静默退化成 `softmax(scores)`。因为所有元素同时减 1e9，
        这个常数在 max-shift 里相互抵消了，mask 等于没加，
        于是得到一个看似正常、实则毫无意义的分布。
        方案 B 显式返回**全 0 权重**（context 为 0 向量），
        语义是"没有可注意的地方"，一眼就能看出不对劲。

    方案 B 需要注意的坑：`-inf` 会让 `max()` 也变成 `-inf`，进而
    `-inf - (-inf) = nan` 污染整批数据。所以下面显式把非有限的 max 换成 0，
    并且用 `np.maximum(denom, 1e-300)` 兜住 0 分母。
    """
    scores = np.asarray(scores, dtype=np.float64)
    if mask is None:
        return softmax(scores, axis=axis)

    valid = np.asarray(mask)
    if valid.ndim == scores.ndim - 1:
        # (B, S) -> (B, 1, S)：必须显式插轴，见上文广播陷阱
        valid = np.expand_dims(valid, axis=-2)
    valid = valid > 0.0

    x = np.where(valid, scores, -np.inf)

    m = x.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)

    denom = e.sum(axis=axis, keepdims=True)
    # denom == 0（整行被 mask）时 0 / 1e-300 == 0，天然得到全 0 权重，不会 nan
    return e / np.maximum(denom, 1e-300)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 log(softmax(x))。

        log_softmax(x)_i = x_i - max(x) - log Σ_j exp(x_j - max(x))

    **为什么不能写成 `np.log(softmax(x))`？** 因为 softmax 会先把很小的概率
    压成 0（下溢），再取 log 就得到 -inf。而这里的写法把"减最大值"和"取对数"
    合并进同一个表达式，全程不需要真的算出那个小概率，因此永远不会下溢。
    交叉熵损失直接建立在 log_softmax 上，这是它数值稳定的根源。
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def softmax_backward(weights: np.ndarray, dweights: np.ndarray, axis: int = -1) -> np.ndarray:
    """softmax 的反向传播。

    设 y = softmax(x)，上游梯度 dL/dy = dweights，则

        dL/dx_i = y_i * ( dweights_i - Σ_j dweights_j * y_j )

    推导要点：
        ∂y_i/∂x_j = y_i (δ_ij - y_j)
        dL/dx_i = Σ_j dL/dy_j * ∂y_j/∂x_i
                = Σ_j dweights_j * y_j (δ_ij - y_i)
                = dweights_i y_i - y_i Σ_j dweights_j y_j
                = y_i ( dweights_i - Σ_j dweights_j y_j )       ∎

    注意括号里那个减项是**该行所有元素共享的标量** ——
    这就是为什么 softmax 的 Jacobian 不是对角的：某个位置的分数变大会压低其它位置。
    也正因为有它，`y_i = 0` 的位置梯度必然为 0（被 mask 的位置不会被"复活"）。
    """
    return weights * (dweights - (dweights * weights).sum(axis=axis, keepdims=True))


# --------------------------------------------------------------------------------------
# 基础层
# --------------------------------------------------------------------------------------
class Linear(Module):
    """仿射变换  y = x @ W + b

        W : (in_dim, out_dim)
        b : (out_dim,)
        x : (..., in_dim)     —— 支持任意前置维度，内部展平成二维做矩阵乘法
        y : (..., out_dim)

    反向传播
    --------
        dW = xᵀ · dy      （把 batch 维求和掉）
        db = Σ dy         （对 batch 维求和）
        dx = dy · Wᵀ

    为什么支持任意前置维度很重要：GRU 里同一层会被喂 (B, in) 的向量，
    注意力里会喂 (B, T, in) 的张量。展平成 (B·T, in) 后一次矩阵乘法搞定，
    比写两套分支干净得多。
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 init_scale: float = 0.08, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.use_bias = bool(bias)
        self._register("W", init_uniform((self.in_dim, self.out_dim), init_scale, rng))
        if self.use_bias:
            self._register("b", np.zeros((self.out_dim,), dtype=np.float64))
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = x @ self._params["W"]
        if self.use_bias:
            y = y + self._params["b"]
        self._cache = {"x": x}
        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")
        x = self._cache["x"]
        dy = np.asarray(dy, dtype=np.float64)

        x2 = x.reshape(-1, self.in_dim)
        dy2 = dy.reshape(-1, self.out_dim)

        self._grads["W"] += x2.T @ dy2
        if self.use_bias:
            self._grads["b"] += dy2.sum(axis=0)
        return dy @ self._params["W"].T


class Embedding(Module):
    """查表：把整数 id 映射成稠密向量。

        weight : (num_embeddings, embedding_dim)
        index  : (...,)        int64
        output : (..., embedding_dim)

    ⚠ 反向传播必须用 `np.add.at`，不能写 `grad[index] += dout`
    -----------------------------------------------------------
    numpy 的**花式索引是拷贝语义**，不是视图：

        grad[idx] += dout          # 错！

    等价于

        tmp = grad[idx]            # 先把 idx 对应的行**复制**出来
        tmp = tmp + dout
        grad[idx] = tmp            # 再写回去

    如果 `idx` 里有重复值（比如源句「2008年11月11日」里 '1' 出现了 4 次，
    或者 batch 里同一个词出现在多句里），写回时**后面的会覆盖前面的**，
    重复出现的那几次梯度就丢了。

    `np.add.at(grad, idx, dout)` 是"无缓冲"累加，会给每个出现位置都加一次。
    scripts/03_encoder.py 第 [2] 节会用具体数字演示这个差别。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 init_scale: float = 0.08, seed: int = 0,
                 padding_idx: int | None = None) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self._register("weight", init_uniform((self.num_embeddings, self.embedding_dim),
                                              init_scale, rng))
        self.padding_idx = padding_idx
        if padding_idx is not None:
            if not 0 <= padding_idx < self.num_embeddings:
                raise ValueError(f"padding_idx={padding_idx} 超出词表范围 [0, {self.num_embeddings})")
            self._params["weight"][padding_idx] = 0.0
        self._cache: dict | None = None

    def forward(self, index) -> np.ndarray:
        # 整数数组索引返回的是**拷贝**，所以外部改不到权重表 —— 正是我们想要的
        index = np.asarray(index, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.num_embeddings):
            raise IndexError(
                f"token id 超出词表范围 [0, {self.num_embeddings})："
                f"min={index.min()}, max={index.max()}"
            )
        out = self._params["weight"][index]
        self._cache = {"index": index}
        return out

    def backward(self, dout: np.ndarray) -> None:
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")
        index = self._cache["index"]
        dout = np.asarray(dout, dtype=np.float64)
        flat_idx = index.reshape(-1)
        flat_dout = dout.reshape(-1, self.embedding_dim)
        np.add.at(self._grads["weight"], flat_idx, flat_dout)
        if self.padding_idx is not None:
            # padding 行永远不该学习：权重和梯度都钉死在 0。
            # 这不是必需的（本项目里 padding 步的输入梯度天然为 0），
            # 而是把「<pad> 不参与任何计算」这条约定在数值上显式化，
            # 免得将来改动编码器时不小心让它收到梯度。
            self._grads["weight"][self.padding_idx] = 0.0
        return None

    def zero_row(self, index: int) -> None:
        """把某一行的权重和梯度都置零（手动工具，等价于事后设置 padding_idx）。"""
        self._params["weight"][index] = 0.0
        self._grads["weight"][index] = 0.0


# --------------------------------------------------------------------------------------
# 数值梯度检验
# --------------------------------------------------------------------------------------
def numerical_gradient_at(
    f: Callable[[], float],
    tensor: np.ndarray,
    flat_indices,
    eps: float = 1e-6,
) -> np.ndarray:
    """对 tensor 中指定的若干（展平后）下标，用**中心差分**估计梯度。

        ∂f/∂x_i ≈ ( f(x_i + ε) - f(x_i - ε) ) / 2ε

    中心差分的截断误差是 O(ε²)，比前向差分的 O(ε) 好得多，
    所以能把 ε 取得稍大一些（1e-6），避开浮点抵消带来的舍入误差。

    `tensor` 会被**就地**修改再还原，因此：
        1. 必须传真正的参数数组，不能传副本；
        2. 必须是 C 连续内存，否则 reshape(-1) 会先拷贝一份，扰动就丢失了；
        3. `f` 每次调用都要重新读取当前的参数值。
    """
    if not tensor.flags["C_CONTIGUOUS"]:
        raise ValueError("numerical_gradient_at 要求 C 连续数组，否则就地扰动会作用在副本上")

    flat = tensor.reshape(-1)
    idx = flat_indices if isinstance(flat_indices, np.ndarray) else np.asarray(list(flat_indices), dtype=np.int64)

    out = np.empty(idx.size, dtype=np.float64)
    for n, i in enumerate(idx):
        i = int(i)
        old = flat[i]
        flat[i] = old + eps
        f_plus = f()
        flat[i] = old - eps
        f_minus = f()
        flat[i] = old
        out[n] = (f_plus - f_minus) / (2.0 * eps)
    return out


def relative_error(a, b) -> float:
    """相对误差 |a-b| / max(|a|+|b|, 1e-12)，取所有元素的最大值。

    判定经验值（float64 + 中心差分）：
        < 1e-7  完美，梯度推导正确
        ~1e-5   可接受，通常是某处数值不稳定
        > 1e-3  基本可以确定公式写错了
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def check_gradient(
    tensors: dict,
    analytic: dict,
    loss_fn: Callable[[], float],
    n_samples: int = 8,
    eps: float = 1e-6,
    rng: np.random.Generator | None = None,
    rel_tol: float = 1e-6,
    noise_safety: float = 10.0,
    spread_eps_factor: float = 4.0,
    meaningful_factor: float = 1e3,
) -> list[dict]:
    """逐张量抽样做梯度检验（自校准判据）。

    参数
    ----
    tensors  : {名字: 数组}。既可以是模型参数，也可以是**输入张量**。
    analytic : {名字: 数组}，解析梯度。
    loss_fn  : 无参函数，返回标量 loss；必须每次重新读取 tensors 的当前值。
    n_samples: 每个张量抽查多少个元素。
    eps      : 中心差分的步长。float64 下推荐 1e-5（见 scripts/02_attention.py 的 ε 扫描）。

    ⚠ 为什么不能用「逐元素相对误差 < 1e-7」这种单一判据
    --------------------------------------------------
    中心差分的舍入噪声下界是 `noise_floor ≈ |L|·ε_mach/(2ε)`。
    实测 `|L|=0.136, ε=1e-5` 时它只有 `1.5e-12`。

    第 4 步就撞上了这个边界：`attn.W_q` 的梯度只有 `5.9e-9`，比 loss 小 8 个数量级
    （因为初始化解码器隐状态高度相似，softmax 的雅可比近于抵消，
    导致注意力打分参数的梯度天然极小）。此时数值梯度**完全被噪声淹没**，
    逐元素相对误差是 `3.5e-2` —— 看起来像 bug，其实完全正确。
    （在梯度最大的元素上，数值估计随 ε 增大收敛到解析值，这是决定性证据。）

    所以本函数改用**自校准**判据：拿数值方法**自身的**不确定度当尺子。

        spread      = |num(ε) - num(4ε)|        数值方法自身的不确定度
        discrepancy = |num(ε) - 解析梯度|        解析与数值的分歧
        通过条件     discrepancy ≤ noise_safety · (spread + rel_tol·|解析|)

    直觉：允许的误差取「数值方法自己的抖动」和「梯度的一个小比例」中较大的那个。

        · 梯度健康（远大于噪声）-> spread 很小，第二项主导 -> 要求**相对精度**
        · 梯度贴着噪声地板      -> spread 主导 -> 只要求「和噪声水平一致」
        · 公式写错             -> discrepancy 与 |梯度| 同量级，
                                  远超两项，**照样会被抓住**

    返回字段里的 `ratio = discrepancy / spread` 是最直观的一个数：
        1 ~ 100   -> 差异完全由数值噪声解释，实现正确
        ≫ 1e3     -> 公式有问题
    （实测：正确实现 ~1e0，故意写错的实现 ~4.6e9。）

    返回
    ----
    [{"name", "shape", "n_checked", "n_meaningful", "max_rel_err", "ratio",
      "max_discrepancy", "spread", "noise_floor", "loss_magnitude",
      "status", "passed", "worst_num", "worst_ana", "flat_index"}, ...]

    status:
        "ok"      通过，且相对误差判据有效（有信噪比足够的元素）
        "low-snr" 通过，但梯度整体贴着噪声地板，相对误差读数没有意义
        "FAIL"    不通过
    """
    rng = rng or np.random.default_rng(0)

    loss_magnitude = abs(float(loss_fn()))
    eps_mach = float(np.finfo(np.float64).eps)
    noise_floor = max(loss_magnitude * eps_mach / (2.0 * eps), 1e-300)

    results: list[dict] = []

    for name, tensor in tensors.items():
        if not tensor.flags["C_CONTIGUOUS"]:
            raise ValueError(f"张量 {name} 不是 C 连续数组")
        n = tensor.size
        k = min(n_samples, n)
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)

        # 用两个不同步长各算一次数值梯度，差值就是数值方法自身的不确定度
        num = numerical_gradient_at(loss_fn, tensor, idx, eps)
        num_alt = numerical_gradient_at(loss_fn, tensor, idx, eps * spread_eps_factor)

        ana_full = np.asarray(analytic[name], dtype=np.float64)
        if ana_full.shape != tensor.shape:
            raise ValueError(f"{name} 的解析梯度形状是 {ana_full.shape}，"
                             f"与张量形状 {tensor.shape} 不符")
        ana = ana_full.reshape(-1)[idx]

        spread = np.abs(num - num_alt)
        discrepancy = np.abs(num - ana)
        # 容差 = 10 · ( 数值方法自身抖动 + 相对精度项 + **噪声地板绝对项** )
        #
        # 最后那个 `noise_floor` 是必须的：如果某个元素同时满足
        # spread = 0 且 |解析梯度| ≈ 0（比如该参数压根没被用到），
        # 前两项都会塌缩到 0，容差变成 1e-300，于是任何舍入级差异
        # （~1e-11）都会被判 FAIL —— 这是**判据的缺陷**，不是公式错误。
        # 加上噪声地板后语义变得明确：梯度低于数值噪声的元素，
        # 只要求"差异不超过噪声水平"，不再要求相对精度。
        tolerance = noise_safety * (spread + rel_tol * np.abs(ana) + noise_floor) + 1e-300

        ok_mask = discrepancy <= tolerance
        passed = bool(np.all(ok_mask))
        j = int(np.argmax(discrepancy / tolerance))

        # 相对误差只有在「信号远大于数值抖动」时才有意义
        meaningful = np.abs(ana) >= meaningful_factor * np.maximum(spread, 1e-300)
        n_meaningful = int(meaningful.sum())
        if n_meaningful:
            rel = discrepancy[meaningful] / np.maximum(
                np.abs(num[meaningful]) + np.abs(ana[meaningful]), 1e-12)
            max_rel = float(rel.max())
        else:
            max_rel = float("nan")

        # spread 偶尔会恰好为 0（两个步长的数值估计逐位相同），
        # 用理论噪声地板兜住分母，避免出现 1e287 这种无意义的比值。
        ratio = float(discrepancy[j] / max(spread[j], noise_floor))

        if not passed:
            status = "FAIL"
        elif n_meaningful:
            status = "ok"
        else:
            status = "low-snr"

        results.append(
            {
                "name": name,
                "shape": tuple(tensor.shape),
                "n_checked": int(k),
                "n_meaningful": n_meaningful,
                "max_rel_err": max_rel,
                "ratio": ratio,
                "max_discrepancy": float(discrepancy[j]),
                "spread": float(spread[j]),
                "noise_floor": noise_floor,
                "loss_magnitude": loss_magnitude,
                "status": status,
                "passed": passed,
                "worst_num": float(num[j]),
                "worst_ana": float(ana[j]),
                "flat_index": int(idx[j]),
            }
        )
    return results


def summarize_gradient_report(results: list[dict]) -> dict:
    """把 `check_gradient` 的结果汇总成一句能说清的话。"""
    failed = [r["name"] for r in results if r["status"] == "FAIL"]
    low_snr = [r["name"] for r in results if r["status"] == "low-snr"]
    rels = [r["max_rel_err"] for r in results if r["max_rel_err"] == r["max_rel_err"]]
    return {
        "passed": not failed,
        "failed": failed,
        "low_snr": low_snr,
        "worst_rel_err": max(rels) if rels else float("nan"),
        "worst_ratio": max((r["ratio"] for r in results), default=0.0),
        "n_tensors": len(results),
        "noise_floor": results[0]["noise_floor"] if results else 0.0,
        "loss_magnitude": results[0]["loss_magnitude"] if results else 0.0,
    }
