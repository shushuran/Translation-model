"""注意力机制：加性（Bahdanau, 2015）与缩放点积（Vaswani, 2017）。

本模块**不依赖任何 RNN**，是一个可以独立验证的纯函数。统一接口：

    context, weights = attn.forward(query, keys, values, mask=None)
    dinputs = attn.backward(dcontext)        # 梯度累加到 attn.grads()，并返回对输入的梯度
    attn.zero_grad()

张量形状约定
-----------
    query   (B, Tq, dq)   查询：解码器当前隐状态。解码时 Tq = 1。
    keys    (B, S,  dk)   键  ：编码器每一步的隐状态序列。S = 源句长度。
    values  (B, S,  dv)   值  ：通常就是 keys 本身（本项目即如此）。
    mask    (B, S) 或可广播形式，1.0 = 有效 token，0.0 = <pad>
    ---------------------------------------------------------------
    weights (B, Tq, S)    注意力权重，每行在有效位置上求和为 1
    context (B, Tq, dv)   加权求和结果，喂给解码器

注意力的本质
-----------
把它当成一次**软查字典**：
    * keys 是字典的"索引"，values 是"释义"；
    * query 是你要查的词；
    * scores 是 query 和每个 key 的匹配程度；
    * softmax 把匹配程度变成概率分布；
    * context 就是"按概率把释义混合起来"。

硬查字典只会取出一个词；软查字典会按权重把所有释义加权平均，
所以它对"查询不完全精确"这件事是可微的 —— 这正是它能被端到端训练的原因。
"""

from __future__ import annotations

import numpy as np

from .nn import Module, init_uniform, softmax_backward, softmax_masked

__all__ = [
    "BahdanauAttention",
    "DotProductAttention",
    "MeanContextAttention",
    "build_attention",
]


def _split_mask(scores: np.ndarray, mask) -> np.ndarray | None:
    """把 0/1 有效性 mask 规整成与 scores 同形状的布尔数组，供反向传播使用。

    为什么要显式插轴：numpy 广播从最右边一维向左对齐。
    如果 mask 是 (B, S) 而 scores 是 (B, Tq, S)，自动广播会把它当成 (1, B, S)：
        · B ≠ Tq 且都 > 1  -> 抛 ValueError
        · Tq == 1 或 B == 1 -> **静默**变成 (B, B, S)
    第二种恰好是解码推理的真实配置（Tq=1, B=32），会悄悄用错样本的 mask。
    所以这里先 expand_dims 成 (B, 1, S)，再广播到 scores 的形状。
    """
    if mask is None:
        return None
    valid = np.asarray(mask)
    if valid.ndim == scores.ndim - 1:
        valid = np.expand_dims(valid, axis=-2)
    return np.broadcast_to(valid > 0.0, scores.shape)


def _check_shapes(query, keys, values, mask) -> None:
    """形状校验。宁可在这里大声报错，也不要让广播缺陷静默传播。"""
    if query.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError(
            f"query/keys/values 都必须是三维 (B, T, d)，"
            f"收到 {query.shape} / {keys.shape} / {values.shape}"
        )
    B, Tq, _ = query.shape
    Bk, S, _ = keys.shape
    Bv, Sv, _ = values.shape
    if not (B == Bk == Bv):
        raise ValueError(f"batch 维不一致：query {B} / keys {Bk} / values {Bv}")
    if S != Sv:
        raise ValueError(f"keys 与 values 的序列长度必须相同，收到 {S} 与 {Sv}")

    if mask is None:
        return
    m = np.asarray(mask)
    if m.ndim == 2:
        if m.shape != (B, S):
            raise ValueError(f"mask 形状应为 (B, S) = ({B}, {S})，收到 {m.shape}")
    elif m.ndim == 3:
        if m.shape != (B, Tq, S):
            raise ValueError(f"mask 形状应为 (B, Tq, S) = ({B}, {Tq}, {S})，收到 {m.shape}")
    else:
        raise ValueError(f"mask 必须是二维 (B,S) 或三维 (B,Tq,S)，收到 {m.ndim} 维")


class BahdanauAttention(Module):
    """加性注意力（Bahdanau et al., 2015，"Neural Machine Translation by Jointly
    Learning to Align and Translate"）。

    打分函数
    --------
        q_proj = W_q · q + b                    (B, Tq, A)
        k_proj = W_k · k + b                    (B, S,  A)
        h      = tanh(q_proj[:, :, None, :] + k_proj[:, None, :, :])   (B, Tq, S, A)
        scores = h · v                          (B, Tq, S)

    为什么需要参数？
        加性注意力先把 query 和 key 各自线性投影到同一个 A 维空间，
        再用 tanh 做非线性、最后用向量 v 投成标量。
        因为引入了 tanh 这一层非线性，它能表达**点积表达不了**的匹配函数
        （点积只能衡量"方向一致性"，是双线性的）。
        代价是多了参数，且计算量更大。

    为什么用 `v` 而不是直接对 h 求和？
        `v` 让不同维度有不同的重要性，这是必要的自由度。
        如果直接 sum(h)，每个维度权重都是 1，表达能力会明显下降。

    关于偏置 b
        这里 q_proj 和 k_proj 各加了一次 b，等价于 tanh 前只有一个偏置 2b
        （因为 tanh(x + 2b) 里 2b 仍是任意常数）。主流实现都这么写。
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int | None = None,
        attn_dim: int = 128,
        init_scale: float = 0.08,
        seed: int = 0,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        self.q_dim = int(q_dim)
        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim) if v_dim is not None else int(k_dim)
        self.attn_dim = int(attn_dim)

        self._register("W_q", init_uniform((self.q_dim, self.attn_dim), init_scale, rng))
        self._register("W_k", init_uniform((self.k_dim, self.attn_dim), init_scale, rng))
        # v 是打分向量的方向，用较小的尺度初始化，避免训练初期 scores 方差过大
        self._register("v", init_uniform((self.attn_dim,), init_scale, rng))
        self._register("b", np.zeros((self.attn_dim,), dtype=np.float64))

        self._cache: dict | None = None
        # 诊断用：记录最近一次前向的中间量，方便调试时查看
        self.last_weights: np.ndarray | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, query, keys, values, mask=None):
        query = np.ascontiguousarray(query, dtype=np.float64)
        keys = np.ascontiguousarray(keys, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        _check_shapes(query, keys, values, mask)

        W_q = self._params["W_q"]
        W_k = self._params["W_k"]
        v = self._params["v"]
        b = self._params["b"]

        # (B, Tq, A) 和 (B, S, A)
        q_proj = query @ W_q + b
        k_proj = keys @ W_k + b

        # 广播相加得到 (B, Tq, S, A)，再过 tanh。
        # 解码时 Tq = 1，这一步就是 (B, 1, S, A) —— 不需要预计算优化就已经很快。
        h = np.tanh(q_proj[:, :, None, :] + k_proj[:, None, :, :])

        # (B, Tq, S)：对最后一个维度（A 维）做加权和
        scores = h @ v

        weights = softmax_masked(scores, mask, axis=-1)
        context = weights @ values

        self._cache = {
            "query": query,
            "keys": keys,
            "values": values,
            "q_proj": q_proj,
            "k_proj": k_proj,
            "h": h,
            "weights": weights,
            "valid": _split_mask(scores, mask),
        }
        self.last_weights = weights
        return context, weights

    # ----------------------------------------------------------------- backward
    def backward(self, dcontext):
        """反向传播。

        返回 {"query", "keys", "values"} 三个输入梯度；
        参数梯度**累加**进 self._grads（配合 zero_grad 使用）。

        梯度链条（从后往前）
        -------------------
            context = weights @ values
                → dweights = dcontext @ valuesᵀ
                → dvalues  = Σ_tq weightsᵀ · dcontext
            weights = softmax(masked_scores)
                → dscores = softmax_backward(weights, dweights)
            scores  = h · v
                → dv = Σ h ⊙ dscores
                → dh = dscores ⊗ v          （外积，恢复成 (B,Tq,S,A)）
            h       = tanh(pre)
                → dpre = dh · (1 - h²)       ← tanh 的导数，必须用前向的 h
            pre     = q_proj ⊕ k_proj（广播相加）
                → dq_proj = Σ_S dpre         （沿 S 求和，把广播"缩回去"）
                → dk_proj = Σ_Tq dpre        （沿 Tq 求和）
            q_proj  = query @ W_q + b
                → dW_q   = queryᵀ · dq_proj
                → dquery = dq_proj @ W_qᵀ
                → db    += Σ dq_proj + Σ dk_proj
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        dcontext = np.asarray(dcontext, dtype=np.float64)

        weights = c["weights"]
        values = c["values"]
        h = c["h"]

        # ---- context = weights @ values ----
        dweights = dcontext @ values.transpose(0, 2, 1)              # (B,Tq,S)
        dvalues = np.einsum("bts,btv->bsv", weights, dcontext)       # (B,S,dv)

        # ---- weights = softmax_masked(scores) ----
        dscores = softmax_backward(weights, dweights, axis=-1)
        # 被 mask 的位置权重恰好是 0，上式已保证其梯度为 0；
        # 这里再乘一次 valid 是**防御性**写法，防止将来改动 softmax 实现后静默出错。
        if c["valid"] is not None:
            dscores = dscores * c["valid"]

        # ---- scores = h · v ----
        v = self._params["v"]
        dv = np.einsum("bts,btsa->a", dscores, h)                     # (A,)
        dh = dscores[:, :, :, None] * v[None, None, None, :]           # (B,Tq,S,A)

        # ---- h = tanh(pre) ----
        dpre = dh * (1.0 - h * h)                                     # (B,Tq,S,A)

        # ---- pre = q_proj[:,:,None,:] + k_proj[:,None,:,:] ----
        dq_proj = dpre.sum(axis=2)                                    # (B,Tq,A)  对 S 求和
        dk_proj = dpre.sum(axis=1)                                    # (B,S, A)  对 Tq 求和

        # ---- q_proj = query @ W_q + b ,  k_proj = keys @ W_k + b ----
        query, keys = c["query"], c["keys"]
        W_q, W_k = self._params["W_q"], self._params["W_k"]

        dW_q = np.einsum("bti,btj->ij", query, dq_proj)               # (dq, A)
        dW_k = np.einsum("bsi,bsj->ij", keys, dk_proj)                # (dk, A)
        db = dq_proj.sum(axis=(0, 1)) + dk_proj.sum(axis=(0, 1))      # (A,)

        dquery = dq_proj @ W_q.T                                      # (B,Tq,dq)
        dkeys = dk_proj @ W_k.T                                       # (B,S, dk)

        # 累加参数梯度（BPTT 里同一个模块会被调用很多次）
        self._grads["W_q"] += dW_q
        self._grads["W_k"] += dW_k
        self._grads["v"] += dv
        self._grads["b"] += db

        return {"query": dquery, "keys": dkeys, "values": dvalues}


class DotProductAttention(Module):
    """缩放点积注意力（Vaswani et al., 2017）。

        scores = (q · kᵀ) / √d
        weights = softmax(scores)
        context = weights @ values

    两个关键设计
    -----------
    1. **为什么要除以 √d？**
       假设 q、k 各维独立、均值 0 方差 1，则点积 q·k 的方差是 d。
       d = 128 时点积的标准差就有 √128 ≈ 11.3，数值会很大。
       数值大 → softmax 变成接近 one-hot 的尖峰 → 反向梯度趋近 0 → 训练停滞。
       除以 √d 把方差重新压回 1，softmax 才处在"既不平均也不极端"的可学区间。

    2. **什么时候不需要参数？**
       如果 query 和 key 的维度相同（本项目都是 128），可以完全不做投影：
       `scores = q·kᵀ/√d`，**零参数**。这是它与加性注意力最本质的对比 ——
       点积注意力衡量的是"两个向量方向有多一致"，是一个固定的双线性型，
       没有任何可学参数；加性注意力则通过学习 W_q、W_k、v 来塑造匹配函数。
       实践结论：维度较小或需要强表达匹配函数时用加性；
       维度较大（≥64）、追求速度和显存时用点积。

    3. 当 dq ≠ dk（比如换成双向编码器）或想降低计算量时，
       可以传入 `dim` 打开 W_q / W_k 投影；此时就会产生参数。
    """

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int | None = None,
        dim: int | None = None,
        use_projection: bool | None = None,
        init_scale: float = 0.08,
        seed: int = 0,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        self.q_dim = int(q_dim)
        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim) if v_dim is not None else int(k_dim)
        self.dim = int(dim) if dim is not None else int(q_dim)

        # 自动判断：投影后维度与原始维度都相同 -> 投影是多余的，直接不做
        if use_projection is None:
            use_projection = not (self.dim == self.q_dim == self.k_dim)
        self.use_projection = bool(use_projection)

        if not self.use_projection and self.q_dim != self.k_dim:
            raise ValueError(
                f"不投影的点积注意力要求 q_dim == k_dim，收到 {self.q_dim} 与 {self.k_dim}；"
                f"请传入 dim 打开投影"
            )

        if self.use_projection:
            self._register("W_q", init_uniform((self.q_dim, self.dim), init_scale, rng))
            self._register("W_k", init_uniform((self.k_dim, self.dim), init_scale, rng))

        self.scale = 1.0 / float(np.sqrt(self.dim))
        self._cache: dict | None = None
        self.last_weights: np.ndarray | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, query, keys, values, mask=None):
        query = np.ascontiguousarray(query, dtype=np.float64)
        keys = np.ascontiguousarray(keys, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        _check_shapes(query, keys, values, mask)

        if self.use_projection:
            q = query @ self._params["W_q"]
            k = keys @ self._params["W_k"]
        else:
            q, k = query, keys

        scores = (q @ k.transpose(0, 2, 1)) * self.scale         # (B,Tq,S)
        weights = softmax_masked(scores, mask, axis=-1)
        context = weights @ values

        self._cache = {
            "query": query,
            "keys": keys,
            "values": values,
            "q": q,
            "k": k,
            "weights": weights,
            "valid": _split_mask(scores, mask),
        }
        self.last_weights = weights
        return context, weights

    # ----------------------------------------------------------------- backward
    def backward(self, dcontext):
        """反向传播。返回 {"query", "keys", "values"}，参数梯度累加进 self._grads。

        梯度链条
        --------
            context = weights @ values
                → dweights = dcontext @ valuesᵀ
                → dvalues  = Σ_tq weightsᵀ · dcontext
            weights = softmax(scores)
                → dscores = softmax_backward(weights, dweights)
            scores  = (q · kᵀ) * scale
                → dq = (dscores @ k)    * scale
                → dk = (dscoresᵀ @ q)   * scale
            （若开了投影）q = query @ W_q, k = keys @ W_k
                → dW_q = queryᵀ · dq ,  dquery = dq @ W_qᵀ
                → dW_k = keysᵀ  · dk ,  dkeys  = dk @ W_kᵀ
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        dcontext = np.asarray(dcontext, dtype=np.float64)

        weights, values = c["weights"], c["values"]

        dweights = dcontext @ values.transpose(0, 2, 1)              # (B,Tq,S)
        dvalues = np.einsum("bts,btv->bsv", weights, dcontext)       # (B,S,dv)

        dscores = softmax_backward(weights, dweights, axis=-1)
        if c["valid"] is not None:
            dscores = dscores * c["valid"]

        q, k = c["q"], c["k"]
        dq = (dscores @ k) * self.scale                              # (B,Tq,dim)
        dk = (dscores.transpose(0, 2, 1) @ q) * self.scale           # (B,S, dim)

        if self.use_projection:
            query, keys = c["query"], c["keys"]
            W_q, W_k = self._params["W_q"], self._params["W_k"]
            self._grads["W_q"] += np.einsum("bti,btj->ij", query, dq)
            self._grads["W_k"] += np.einsum("bsi,bsj->ij", keys, dk)
            dquery = dq @ W_q.T
            dkeys = dk @ W_k.T
        else:
            dquery, dkeys = dq, dk

        return {"query": dquery, "keys": dkeys, "values": dvalues}


# --------------------------------------------------------------------------------------
# 无注意力基线
# --------------------------------------------------------------------------------------
class MeanContextAttention(Module):
    """「无注意力」基线：context 恒等于 values 的**掩码加权平均**。

        w[b, j]  = mask[b, j] / Σ_j mask[b, j]        （与 query 无关，所有解码步都一样）
        context  = w @ values

    这是 Bahdanau 等人提出注意力**之前**的标准做法：把整个源句压成一个
    **固定长度**的 context 向量，解码的每一步都用同一个。
    等价于「注意力权重恒为均匀分布」—— 所以它和真实注意力的对比
    就是"动态对齐 vs 固定摘要"的对比。

    它没有可训练参数，前向反向都极简：

        forward   context = w @ values
        backward  dvalues = Σ_t w ⊗ dcontext_t      （沿解码步求和）
                  dquery = dkeys = 0                （query/keys 压根没参与）

    保留 `forward` 的完整签名和返回值，是为了让它能**无缝替换**真实注意力 ——
    这样消融实验里唯一的变量就是"用不用注意力"，其他代码一行都不用改。
    """

    def __init__(self, q_dim: int, k_dim: int, v_dim: int | None = None, **kwargs) -> None:
        super().__init__()
        self.q_dim = int(q_dim)
        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim) if v_dim is not None else int(k_dim)
        self._cache: dict | None = None
        self.last_weights: np.ndarray | None = None

    def forward(self, query, keys, values, mask=None):
        query = np.ascontiguousarray(query, dtype=np.float64)
        keys = np.ascontiguousarray(keys, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float64)
        _check_shapes(query, keys, values, mask)

        B, Tq, _ = query.shape
        S = values.shape[1]

        if mask is None:
            valid = np.ones((B, S), dtype=np.float64)
        else:
            valid = (np.asarray(mask) > 0).astype(np.float64)
        w = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1e-300)     # (B, S)

        # 与 query 无关：每个解码步拿到的是同一份权重
        weights = np.ascontiguousarray(np.broadcast_to(w[:, None, :], (B, Tq, S)))
        context = weights @ values

        self._cache = {"query": query, "keys": keys, "values": values, "w": w}
        self.last_weights = weights
        return context, weights

    def backward(self, dcontext):
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")
        c = self._cache
        dcontext = np.asarray(dcontext, dtype=np.float64)

        dvalues = np.einsum("bs,btv->bsv", c["w"], dcontext)
        return {
            "query": np.zeros_like(c["query"]),
            "keys": np.zeros_like(c["keys"]),
            "values": dvalues,
        }


# --------------------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------------------
def build_attention(cfg, q_dim: int, k_dim: int, v_dim: int | None = None) -> Module:
    """按 ModelConfig.attn_type 造一个注意力模块。

    cfg.attn_type:
        "bahdanau" -> BahdanauAttention（加性，带参数）
        "dot"      -> DotProductAttention（点积；q_dim == k_dim 时零参数）
        "mean"     -> MeanContextAttention（**无注意力基线**：固定 context 向量）
    """
    if cfg.attn_type == "bahdanau":
        return BahdanauAttention(
            q_dim=q_dim, k_dim=k_dim, v_dim=v_dim,
            attn_dim=cfg.attn_dim, init_scale=cfg.init_scale,
        )
    if cfg.attn_type == "dot":
        return DotProductAttention(
            q_dim=q_dim, k_dim=k_dim, v_dim=v_dim,
            dim=None if q_dim == k_dim else cfg.attn_dim,
            init_scale=cfg.init_scale,
        )
    if cfg.attn_type == "mean":
        return MeanContextAttention(q_dim=q_dim, k_dim=k_dim, v_dim=v_dim)
    raise ValueError(
        f"未知的 attn_type: {cfg.attn_type!r}（应为 'bahdanau' / 'dot' / 'mean'）"
    )
