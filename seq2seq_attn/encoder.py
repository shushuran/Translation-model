"""编码器：Embedding + 单向 GRU，含完整的手写反向传播（BPTT）。

    src_ids (B, S) ──Embedding──▶ (B, S, D) ──逐时间步 GRU──▶ (B, S, H)
                                                                 │
                                                                 ├─▶ H_seq  给第 4 步的注意力当 keys/values
                                                                 └─▶ h_final 给第 4 步当解码器初始隐状态

GRU 公式约定：**PyTorch 约定**（与 `torch.nn.GRU` 完全一致）

    r = σ(x·W_ir + b_ir + h·W_hr + b_hr)          # 重置门 reset
    z = σ(x·W_iz + b_iz + h·W_hz + b_hz)          # 更新门 update
    n = tanh(x·W_in + b_in + r ⊙ (h·W_hn + b_hn)) # 候选状态 new
    h' = (1 - z) ⊙ n + z ⊙ h

注意两点容易混淆的地方：

1. **z 的含义**。PyTorch 里 `z` 是"**保留旧状态**的比例"：
   `z = 1` 时 `h' = h`（完全不更新），`z = 0` 时 `h' = n`（完全采纳新候选）。
   而 Cho et al. 2014 原论文写的是 `h' = (1-z)⊙h + z⊙h~`，那里的 `z` 是
   "采纳新值"的比例 —— **两者含义正好相反**。查资料时务必先看公式而不是看符号名。

2. **r 的位置**。PyTorch 把 `r` 乘在**隐藏态的投影**上：`r ⊙ (h·W_hn + b_hn)`；
   原论文写的是 `U(r ⊙ h)`，即先对 h 逐元素缩放再做线性变换。
   数学上不等价，实现时必须选一种。
"""

from __future__ import annotations

import numpy as np

from .nn import Embedding, Module, init_uniform, sigmoid

__all__ = ["GRUCell", "GRUEncoder"]

# 三个门在拼接矩阵里的列顺序。与 PyTorch 的 weight_ih 保持一致（r, z, n）。
GATE_ORDER = ("r", "z", "n")


class GRUCell(Module):
    """单步 GRU。

        x : (B, input_dim)
        h : (B, hidden_dim)
        h': (B, hidden_dim)

    **工程实现要点：三个门拼成一个矩阵，一次矩阵乘法算完。**

    朴素写法是 6 个独立的线性层（r/z/n 各一个 input 投影和一个 hidden 投影），
    要做 6 次矩阵乘法。这里把列方向拼起来：

        W_ih : (input_dim, 3·hidden_dim)      列顺序 [r | z | n]
        W_hh : (hidden_dim, 3·hidden_dim)
        b_ih, b_hh : (3·hidden_dim,)

        gates_x = x @ W_ih + b_ih     # (B, 3H)
        gates_h = h @ W_hh + b_hh     # (B, 3H)

    一次矩阵乘法就得到全部三个门的输入部分，再切片即可。
    这不只是写法简洁 —— 大矩阵乘法能充分利用 BLAS，实测比 6 次小乘法快得多。
    PyTorch 内部就是这么存的（`weight_ih_l0` 的形状是 `(3H, input_dim)`）。

    唯一需要小心的是：反向传播时也要把切片后的梯度**按同样的顺序拼回去**。
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 init_scale: float = 0.08, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_dim = 3 * self.hidden_dim

        self._register("W_ih", init_uniform((self.input_dim, self.gate_dim), init_scale, rng))
        self._register("W_hh", init_uniform((self.hidden_dim, self.gate_dim), init_scale, rng))
        self._register("b_ih", np.zeros((self.gate_dim,), dtype=np.float64))
        self._register("b_hh", np.zeros((self.gate_dim,), dtype=np.float64))

        self._cache: dict | None = None

    # ------------------------------------------------------------------ 工具
    def _split(self, gates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把 (B, 3H) 按 [r | z | n] 切成三份 (B, H)。"""
        H = self.hidden_dim
        return gates[:, :H], gates[:, H : 2 * H], gates[:, 2 * H :]

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float64)
        h = np.ascontiguousarray(h, dtype=np.float64)

        W_ih, W_hh = self._params["W_ih"], self._params["W_hh"]
        gates_x = x @ W_ih + self._params["b_ih"]        # (B, 3H)
        gates_h = h @ W_hh + self._params["b_hh"]        # (B, 3H)
        gx_r, gx_z, gx_n = self._split(gates_x)
        gh_r, gh_z, gh_n = self._split(gates_h)

        r = sigmoid(gx_r + gh_r)                         # (B, H)
        z = sigmoid(gx_z + gh_z)                         # (B, H)
        n = np.tanh(gx_n + r * gh_n)                     # (B, H)

        h_new = (1.0 - z) * n + z * h                    # (B, H)

        self._cache = {
            "x": x, "h": h,
            "gh_n": gh_n, "r": r, "z": z, "n": n,
        }
        return h_new

    # ----------------------------------------------------------------- backward
    def backward(self, dh_new: np.ndarray):
        """给定上游梯度 dh_new，返回 {"x": dx, "h": dh_prev}，参数梯度累加进 self._grads。

        梯度链条（从 h_new 往前推）
        -------------------------
            h_new = (1 - z) ⊙ n + z ⊙ h
                → dn         = dh_new ⊙ (1 - z)
                → dz         = dh_new ⊙ (h - n)      ← 注意 ∂/∂z 是 (h - n)，不是 h
                → dh(直接路) = dh_new ⊙ z

            n = tanh(gx_n + r ⊙ gh_n)
                → dpre_n = dn ⊙ (1 - n²)             ← tanh 的导数，用前向缓存的 n
                → dgx_n  = dpre_n
                → d(r⊙gh_n) = dpre_n
                → dr     = dpre_n ⊙ gh_n
                → dgh_n  = dpre_n ⊙ r

            r = sigmoid(gx_r + gh_r)
                → dpre_r = dr ⊙ r ⊙ (1 - r)          ← sigmoid 的导数
                → dgx_r = dgh_r = dpre_r
            z = sigmoid(gx_z + gh_z)
                → dpre_z = dz ⊙ z ⊙ (1 - z)
                → dgx_z = dgh_z = dpre_z

            gates_x = x @ W_ih + b_ih   ,   gates_h = h @ W_hh + b_hh
                → 把 [dpre_r | dpre_z | dpre_n] 拼回 (B, 3H)（顺序必须和 forward 一致！）
                → dW_ih = xᵀ · d_gates_x ,  db_ih = Σ d_gates_x ,  dx = d_gates_x @ W_ihᵀ
                → dW_hh = hᵀ · d_gates_h ,  db_hh = Σ d_gates_h ,  dgh  = d_gates_h @ W_hhᵀ
                → dh_prev = dgh + dh(直接路)
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        x, h = c["x"], c["h"]
        r, z, n, gh_n = c["r"], c["z"], c["n"], c["gh_n"]
        dh_new = np.asarray(dh_new, dtype=np.float64)

        # ---- h_new = (1-z)⊙n + z⊙h ----
        dn = dh_new * (1.0 - z)
        dz = dh_new * (h - n)
        dh_direct = dh_new * z

        # ---- n = tanh(gx_n + r ⊙ gh_n) ----
        dpre_n = dn * (1.0 - n * n)
        dr = dpre_n * gh_n
        dgh_n = dpre_n * r

        # ---- 两个 sigmoid 门 ----
        dpre_r = dr * r * (1.0 - r)
        dpre_z = dz * z * (1.0 - z)

        # ---- 按 [r | z | n] 顺序拼回 (B, 3H) ----
        d_gates_x = np.concatenate([dpre_r, dpre_z, dpre_n], axis=-1)
        d_gates_h = np.concatenate([dpre_r, dpre_z, dgh_n], axis=-1)

        W_ih, W_hh = self._params["W_ih"], self._params["W_hh"]
        self._grads["W_ih"] += x.T @ d_gates_x
        self._grads["W_hh"] += h.T @ d_gates_h
        self._grads["b_ih"] += d_gates_x.sum(axis=0)
        self._grads["b_hh"] += d_gates_h.sum(axis=0)

        dx = d_gates_x @ W_ih.T
        dh_prev = d_gates_h @ W_hh.T + dh_direct
        return {"x": dx, "h": dh_prev}

    def num_gates(self) -> int:
        return len(GATE_ORDER)


class GRUEncoder(Module):
    """Embedding + 单向 GRU 编码器。

    padding 的处理方式是本类最关键的细节
    ------------------------------------
    朴素做法是直接在补齐后的序列上跑 GRU。因为本项目是**右填充**，
    padding 都在句尾，看起来"不影响前面的输出"。但有两个问题：

        1. `h_final` 会被 padding 污染。句尾跑了几步 `<pad>` 之后，
           最终隐状态已经偏离"读完最后一个真词"的状态了。
           而第 4 步要用 `h_final` 当解码器的初始隐状态 —— 直接影响翻译质量。
        2. 不同 batch 里 padding 的个数不同，同一个句子在不同 batch 中
           会得到**不一样**的 `h_final`，结果随 batch 组成而漂移，无法复现。

    解决办法：**让状态在 padding 位置冻结**，即

        h_t = valid_t · GRU(x_t, h_{t-1}) + (1 - valid_t) · h_{t-1}

    `valid_t` 就是 `src_mask[:, t]`。这样 padding 步等价于"跳过"，
    编码结果与 batch 组成、padding 多少完全无关。
    scripts/03_encoder.py 第 [7] 节会实测两种做法的差别。
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 init_scale: float = 0.08, seed: int = 0, pad_id: int | None = None,
                 zero_pad_embedding: bool = False) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.pad_id = pad_id

        self.emb = self.register_module(
            "emb",
            Embedding(vocab_size, emb_dim, init_scale, seed=seed,
                      padding_idx=pad_id if zero_pad_embedding else None),
        )
        self.cell = self.register_module(
            "cell", GRUCell(emb_dim, hidden_dim, init_scale, seed=seed + 1)
        )

        self._cache: dict | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, src, src_mask):
        """src: (B, S) int64 索引；src_mask: (B, S) 1=真实 0=padding。

        返回 (H_seq, h_final)：
            H_seq   (B, S, H)  每个位置的隐状态（padding 位置是冻结的旧状态）
            h_final (B, H)     读完最后一个真词后的隐状态
        """
        src = np.asarray(src, dtype=np.int64)
        src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
        if src.ndim != 2:
            raise ValueError(f"src 必须是二维 (B, S)，收到 {src.shape}")
        if src_mask.shape != src.shape:
            raise ValueError(f"src_mask 形状必须与 src 一致，收到 {src_mask.shape} 与 {src.shape}")

        B, S = src.shape
        x = self.emb.forward(src)                      # (B, S, D)

        h = np.zeros((B, self.hidden_dim), dtype=np.float64)
        hs = np.empty((B, S, self.hidden_dim), dtype=np.float64)
        # ⚠ 必须逐步保存 cell 的中间量。cell._cache 只有一个槽位，
        #   循环里调用 S 次之后只剩最后一次的，反向传播会全部用错。
        cell_caches = []
        for t in range(S):
            h_new = self.cell.forward(x[:, t, :], h)   # (B, H)
            cell_caches.append(self.cell.save_cache())
            valid = src_mask[:, t][:, None]            # (B, 1)
            h = valid * h_new + (1.0 - valid) * h      # padding 位置冻结状态
            hs[:, t, :] = h

        self._cache = {"src": src, "src_mask": src_mask, "S": S, "cell_caches": cell_caches}
        return hs, hs[:, S - 1, :].copy()

    # ----------------------------------------------------------------- backward
    def backward(self, dH_seq, dh_final=None):
        """沿时间反向传播（BPTT）。

        dH_seq  : (B, S, H) 对每个位置输出的梯度
        dh_final: (B, H)   对 h_final 的梯度（可选；等价于加到 dH_seq[:, -1] 上）

        返回 None（编码器的输入是整数 id，本身没有梯度）。
        参数梯度全部累加进各子模块的梯度缓冲区。

        **BPTT 的三个关键点**

        1. **同一个 cell 被调用 S 次，参数梯度必须累加。**
           所以循环里每次 `cell.backward()` 都是往上加 ——
           第 2 步的 `Module` 就是按"backward 累加"设计的，这里直接用。
           相应地，**调用方**必须先 `encoder.zero_grad()`；
           `backward()` 自己不清零（与 PyTorch 的约定一致），
           这样同一个模块才能被多条路径复用。

        2. **梯度沿时间反向流动。** 第 t 步的 `dh` 有两个来源：
           来自输出 `dH_seq[:, t]`，以及从第 t+1 步传回来的 `dh_carry`。

        3. **冻结状态的反向。** 前向是 `h_t = valid·h_new + (1-valid)·h_{t-1}`，
           所以梯度也分两路：
               `valid·dh_t`       -> 走 cell 内部，回到 `h_{t-1}`
               `(1-valid)·dh_t`   -> **直接**回到 `h_{t-1}`（跳过 cell）
           漏掉第二条会在 padding 位置丢梯度。

        4. **必须逐步恢复 cell 的中间量。** 前向循环调用了 cell S 次，
           而 `cell._cache` 只有一个槽位，最后只剩第 S-1 步的。
           反向循环里每次 backward 之前都要 `cell.load_cache(cell_caches[t])`，
           否则每个时间步都在用错误的中间量算梯度 —— 形状全对、不报错，
           但梯度完全错（本项目实测相对误差 1.0）。
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        src_mask, S = c["src_mask"], c["S"]
        cell_caches = c["cell_caches"]
        dH_seq = np.asarray(dH_seq, dtype=np.float64)
        if dH_seq.shape != (c["src"].shape[0], S, self.hidden_dim):
            raise ValueError(f"dH_seq 形状应为 {(c['src'].shape[0], S, self.hidden_dim)}，"
                             f"收到 {dH_seq.shape}")

        dhs = dH_seq.copy()
        if dh_final is not None:
            dhs[:, S - 1, :] += np.asarray(dh_final, dtype=np.float64)

        # 注意：这里**不**调用 zero_grad()。梯度逐时间步累加到子模块的缓冲区里，
        # 清零是调用方的责任（先 encoder.zero_grad() 再 encoder.backward()），
        # 这样同一个 encoder 才能被多条路径复用。
        # 对照实验里那个"忘记累加"的 bug，就是在循环内部错误地清零。
        dx_seq = np.zeros((dH_seq.shape[0], S, self.emb_dim), dtype=np.float64)
        dh_carry = np.zeros((dH_seq.shape[0], self.hidden_dim), dtype=np.float64)

        for t in reversed(range(S)):
            dh_t = dhs[:, t, :] + dh_carry
            valid = src_mask[:, t][:, None]            # (B, 1)

            dh_new = dh_t * valid                      # 走 cell 的那一路
            dh_skip = dh_t * (1.0 - valid)             # 直接跳过 cell 的那一路

            self.cell.load_cache(cell_caches[t])       # ← 恢复第 t 步的中间量
            dinputs = self.cell.backward(dh_new)
            dx_seq[:, t, :] = dinputs["x"]
            dh_carry = dinputs["h"] + dh_skip

        # 逐位置梯度一次性回传给 Embedding（内部用 np.add.at 处理重复索引）
        self.emb.backward(dx_seq)
        return None

    def num_params_breakdown(self) -> dict:
        return {
            "embedding": self.emb.num_params(),
            "gru": self.cell.num_params(),
            "total": self.num_params(),
        }
