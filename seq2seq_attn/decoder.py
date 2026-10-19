"""带注意力机制的解码器（Bahdanau 式，前向 + 反向）。

架构
----
    桥接：  s₀ = tanh(h_final · W_b + b_b)                 把编码器末状态变成解码器起点

    第 t 步（t = 0 … T-1）：
        c_t, w_t = attention(query = s_{t-1}, keys = H_seq, values = H_seq, mask = src_mask)
        x_t      = [ emb(y_{t-1}) ; c_t ]                    context 拼进 GRU 的输入
        s_t      = GRU(x_t, s_{t-1})
        logits_t = W_o · [ s_t ; c_t ]                       输出层

    `y_{t-1}` 就是 `tgt_in[t]`（数据管道已经做好 `<bos> + y[:-1]` 的错位）。

为什么 attention 的 query 用 s_{t-1} 而不是 s_t
---------------------------------------------
这正是 **Bahdanau 式** 与 **Luong 式** 的分水岭：

    Bahdanau (2015)：先用 s_{t-1} 算 context，再把 context 喂进 GRU
                     -> 注意力参与"状态更新"，context 影响循环本身
    Luong    (2015)：先算 s_t，再用 s_t 算 context，context 只用在输出层
                     -> 注意力只参与"输出"，不影响循环

Bahdanau 式配加性注意力，Luong 式配点积注意力，这是本项目的默认配对。
两者代码差别只有"attn 在 cell 之前还是之后"，但表达能力和训练动态并不相同。

反向传播要多做两件事
------------------
1. 第 t 步里 `context` 被用在**两个地方**（GRU 输入 + 输出层），
   `s_t` 也被用在**两个地方**（本步输出层 + 下一步的 GRU/注意力）。
   所以每一步都要把不同路径来的梯度**相加**。
2. 注意力的 `dkeys` 和 `dvalues` **相加**后回传给编码器 —— 因为本项目传的是
   `values = keys = H_seq`，两者本来就是同一份张量。

⚠ 单槽位缓存的坑（第 3 步踩过）——本模块里有**三个**模块中招
--------------------------------------------------------
`Module._cache` 只有一个槽位，每次 `forward` 都覆盖。解码循环里：

    self.attn   被调用 T 次   -> 必须逐步 save_cache       ✅ 已处理
    self.cell   被调用 T 次   -> 必须逐步 save_cache       ✅ 已处理
    self.out    被调用 T 次   -> 必须逐步 save_cache       ✅ 已处理 ← 最容易漏！
    self.emb    被调用 1 次   （整段 tgt_in 一次查完）     不需要
    self.bridge 被调用 1 次                              不需要

`self.out` 这一个是写代码时最容易忽略的：它看起来是个"无状态的投影层"，
但在 teacher forcing 下每个时间步的输入 `o_t` 都不同，反向时必须回到对应的那一次。
"""

from __future__ import annotations

import numpy as np

from .attention import build_attention
from .encoder import GRUCell
from .nn import Embedding, Linear, Module

__all__ = ["AttentionDecoder", "OUTPUT_MODES", "INIT_MODES"]

OUTPUT_MODES = ("s_c", "s_c_emb", "s")
INIT_MODES = ("bridge", "direct", "zero")


class AttentionDecoder(Module):
    """基于 GRU + 注意力（Bahdanau 式）的解码器。

    结构由两个配置项决定，方便第 8 步做消融：

        output_mode : 输出层吃什么      "s_c" | "s_c_emb" | "s"
        init_mode   : 初始隐状态从哪来  "bridge" | "direct" | "zero"
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        src_hidden_dim: int,
        attn_cfg,
        output_mode: str = "s_c",
        init_mode: str = "bridge",
        init_scale: float = 0.08,
        seed: int = 0,
        pad_id: int | None = None,
    ) -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode 必须是 {OUTPUT_MODES} 之一，收到 {output_mode!r}")
        if init_mode not in INIT_MODES:
            raise ValueError(f"init_mode 必须是 {INIT_MODES} 之一，收到 {init_mode!r}")

        self.vocab_size = int(vocab_size)
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.src_hidden_dim = int(src_hidden_dim)
        self.output_mode = output_mode
        self.init_mode = init_mode
        self.pad_id = pad_id

        self.emb = self.register_module(
            "emb", Embedding(vocab_size, emb_dim, init_scale, seed=seed, padding_idx=pad_id)
        )
        self.attn = self.register_module(
            "attn",
            build_attention(attn_cfg, q_dim=hidden_dim,
                            k_dim=src_hidden_dim, v_dim=src_hidden_dim),
        )
        # 解码 GRU 的输入是 [嵌入 ; context]
        self.cell = self.register_module(
            "cell", GRUCell(emb_dim + src_hidden_dim, hidden_dim, init_scale, seed=seed + 1)
        )
        self.bridge = self.register_module(
            "bridge", Linear(src_hidden_dim, hidden_dim, init_scale=init_scale, seed=seed + 2)
        )
        self.out = self.register_module(
            "out",
            Linear(self.output_input_dim(), vocab_size, init_scale=init_scale, seed=seed + 3),
        )
        self._cache: dict | None = None

    # ------------------------------------------------------------------ 结构工具
    def output_input_dim(self) -> int:
        if self.output_mode == "s":
            return self.hidden_dim
        if self.output_mode == "s_c":
            return self.hidden_dim + self.src_hidden_dim
        return self.hidden_dim + self.src_hidden_dim + self.emb_dim

    def output_logit_slices(self) -> dict:
        """输出层输入各段的切片范围，便于调试与可视化。"""
        H, Hs, E = self.hidden_dim, self.src_hidden_dim, self.emb_dim
        if self.output_mode == "s":
            return {"s": (0, H), "ctx": None, "emb": None}
        if self.output_mode == "s_c":
            return {"s": (0, H), "ctx": (H, H + Hs), "emb": None}
        return {"s": (0, H), "ctx": (H, H + Hs), "emb": (H + Hs, H + Hs + E)}

    def _build_output_input(self, s_new, ctx, emb_t):
        if self.output_mode == "s":
            return s_new
        if self.output_mode == "s_c":
            return np.concatenate([s_new, ctx], axis=-1)
        return np.concatenate([s_new, ctx, emb_t], axis=-1)

    def _split_output_grad(self, dout):
        """把输出层输入梯度拆回 (ds, dctx, demb)；demb 为 None 表示该模式没用到嵌入。"""
        H, Hs, E = self.hidden_dim, self.src_hidden_dim, self.emb_dim
        if self.output_mode == "s":
            return dout, np.zeros((dout.shape[0], Hs), dtype=np.float64), None
        if self.output_mode == "s_c":
            return dout[:, :H], dout[:, H:H + Hs], None
        return dout[:, :H], dout[:, H:H + Hs], dout[:, H + Hs:H + Hs + E]

    def initial_state(self, src_h_final):
        """按 init_mode 产生解码器起点 s₀，同时返回反向所需的中间量（可能为 None）。"""
        if self.init_mode == "zero":
            s0 = np.zeros((src_h_final.shape[0], self.hidden_dim), dtype=np.float64)
            return s0, None
        if self.init_mode == "direct":
            if src_h_final.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"init_mode='direct' 要求编码器与解码器隐维度相同，"
                    f"收到 {src_h_final.shape[1]} 与 {self.hidden_dim}"
                )
            return np.ascontiguousarray(src_h_final, dtype=np.float64), None
        pre = self.bridge.forward(src_h_final)
        s0 = np.tanh(pre)
        return s0, {"s0": s0}

    # ------------------------------------------------------------------ forward
    def forward(self, tgt_in, src_h, src_mask, src_h_final, init_state=None):
        """教师强制（teacher forcing）下的完整解码。

        参数
        ----
        tgt_in      : (B, T)     int64   解码器输入 = `<bos>` + y[:-1]
        src_h       : (B, S, Hs)         编码器隐状态序列（同时当 keys 和 values）
        src_mask    : (B, S)             1=真实 0=padding
        src_h_final : (B, Hs)            编码器末状态（桥接用）
        init_state  : (B, H) 或 None     显式指定 s_{-1}；给定时跳过桥接，
                                         供第 6 步的逐步推理复用

        返回
        ----
        logits  : (B, T, V)
        weights : (B, T, S)    每个解码步的注意力权重（第 7 步画热力图要用）
        """
        tgt_in = np.asarray(tgt_in, dtype=np.int64)
        src_h = np.ascontiguousarray(src_h, dtype=np.float64)
        src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
        src_h_final = np.ascontiguousarray(src_h_final, dtype=np.float64)
        if tgt_in.ndim != 2:
            raise ValueError(f"tgt_in 必须是二维 (B, T)，收到 {tgt_in.shape}")
        if src_h.ndim != 3:
            raise ValueError(f"src_h 必须是三维 (B, S, Hs)，收到 {src_h.shape}")
        if src_mask.shape != src_h.shape[:2]:
            raise ValueError(f"src_mask 形状应为 {src_h.shape[:2]}，收到 {src_mask.shape}")

        B, T = tgt_in.shape
        S = src_h.shape[1]
        external_init = init_state is not None

        if external_init:
            s = np.ascontiguousarray(init_state, dtype=np.float64)
            bridge_cache = None
        else:
            s, bridge_cache = self.initial_state(src_h_final)

        x_emb = self.emb.forward(tgt_in)                    # (B, T, E)
        emb_seq = x_emb

        logits = np.empty((B, T, self.vocab_size), dtype=np.float64)
        weights = np.empty((B, T, S), dtype=np.float64)

        # ⚠ 逐步保存。cell / attn / out 都被调用 T 次，三者都要保存。
        cell_caches: list = []
        attn_caches: list = []
        out_caches: list = []

        for t in range(T):
            # 1) 注意力：query = s_{t-1}（Bahdanau 式）
            ctx3, w3 = self.attn.forward(s[:, None, :], src_h, src_h, src_mask)
            ctx = ctx3[:, 0, :]                             # (B, Hs)
            weights[:, t, :] = w3[:, 0, :]
            attn_caches.append(self.attn.save_cache())

            # 2) GRU：输入 = [emb(y_{t-1}) ; context]
            emb_t = emb_seq[:, t, :]
            x_t = np.concatenate([emb_t, ctx], axis=-1)
            s_new = self.cell.forward(x_t, s)
            cell_caches.append(self.cell.save_cache())

            # 3) 输出层
            o_t = self._build_output_input(s_new, ctx, emb_t)
            logits[:, t, :] = self.out.forward(o_t)
            out_caches.append(self.out.save_cache())

            s = s_new

        self._cache = {
            "tgt_in": tgt_in,
            "src_mask": src_mask,
            "B": B, "T": T, "S": S,
            "bridge_cache": bridge_cache,
            "external_init": external_init,
            "cell_caches": cell_caches,
            "attn_caches": attn_caches,
            "out_caches": out_caches,
        }
        return logits, weights

    # ----------------------------------------------------------------- backward
    def backward(self, dlogits):
        """反向传播。参数梯度累加进各子模块，并返回对输入的梯度。

        dlogits : (B, T, V)

        返回
        ----
        {"src_h": (B,S,Hs), "src_h_final": (B,Hs)}
        若 forward 时传了 `init_state`，额外返回 `{"init_state": (B,H)}`，
        且 `src_h_final` 为 0（那条路径不存在）。

        ⚠ 梯度**累加**，不清零。调用方必须先 `decoder.zero_grad()`，
        再 `decoder.backward(...)`。这与 PyTorch 的约定一致，
        也让同一个解码器能被多条路径复用（比如后面接 Beam Search 的训练）。

        梯度链条（逐时间步倒着走）
        ------------------------
            do_t = out.backward(dlogits_t)                  (B, 输出层输入维)
            ds_t = ds_carry + do_t[s 段]                     ← 下一步 + 本步输出层
            dctx = do_t[c 段]                                ← 输出层那一路

            cell:  s_t = cell([emb_t ; ctx], s_{t-1})
                dcell   = cell.backward(ds_t)
                demb_t  = dcell.x[:E]
                dctx   += dcell.x[E:]                        ← GRU 输入那一路（相加！）
                ds_prev = dcell.h

            attn:  c_t = attention(query=s_{t-1}, keys=src_h, values=src_h)
                dattn   = attn.backward(dctx)
                ds_prev += dattn.query                       ← query 就是 s_{t-1}
                dsrc_h += dattn.keys + dattn.values          ← 同一份张量，相加！
        """
        if self._cache is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")

        c = self._cache
        B, T, S = c["B"], c["T"], c["S"]
        Hs, E = self.src_hidden_dim, self.emb_dim
        dlogits = np.asarray(dlogits, dtype=np.float64)
        if dlogits.shape != (B, T, self.vocab_size):
            raise ValueError(f"dlogits 形状应为 {(B, T, self.vocab_size)}，收到 {dlogits.shape}")

        # 注意：不清零。梯度逐时间步累加到各子模块（与 PyTorch 约定一致），
        # 清零由调用方负责：先 decoder.zero_grad() 再 decoder.backward(...)。
        dsrc_h = np.zeros((B, S, Hs), dtype=np.float64)
        demb_seq = np.zeros((B, T, E), dtype=np.float64)
        ds_carry = np.zeros((B, self.hidden_dim), dtype=np.float64)

        for t in reversed(range(T)):
            # ---- 输出层 ----
            self.out.load_cache(c["out_caches"][t])
            do_t = self.out.backward(dlogits[:, t, :])
            ds_t, dctx, demb_from_out = self._split_output_grad(do_t)
            ds_t = ds_t + ds_carry
            if demb_from_out is not None:
                demb_seq[:, t, :] += demb_from_out

            # ---- GRU cell ----
            self.cell.load_cache(c["cell_caches"][t])
            dcell = self.cell.backward(ds_t)
            dx_t = dcell["x"]                                # (B, E + Hs)
            demb_seq[:, t, :] += dx_t[:, :E]
            dctx = dctx + dx_t[:, E:]
            ds_prev = dcell["h"]

            # ---- 注意力 ----
            self.attn.load_cache(c["attn_caches"][t])
            dattn = self.attn.backward(dctx[:, None, :])
            ds_prev = ds_prev + dattn["query"][:, 0, :]
            dsrc_h += dattn["keys"] + dattn["values"]

            ds_carry = ds_prev

        # ---- 目标端嵌入 ----
        self.emb.backward(demb_seq)

        # ---- 桥接 ----
        out = {"src_h": dsrc_h}
        if c["external_init"]:
            # s_{-1} 是外部传入的，梯度交回调用方；桥接那条路不存在
            out["init_state"] = ds_carry
            out["src_h_final"] = np.zeros((B, Hs), dtype=np.float64)
            return out

        bc = c["bridge_cache"]
        if bc is not None:
            s0 = bc["s0"]
            dpre = ds_carry * (1.0 - s0 * s0)                # tanh 的导数
            out["src_h_final"] = self.bridge.backward(dpre)
        elif self.init_mode == "direct":
            out["src_h_final"] = ds_carry
        else:                                                # zero
            out["src_h_final"] = np.zeros((B, Hs), dtype=np.float64)
        return out

    def num_params_breakdown(self) -> dict:
        parts = {name: m.num_params() for name, m in self._modules.items()}
        parts["total"] = self.num_params()
        return parts

    # ------------------------------------------------------------------ 推理
    def decode_step(self, prev_token, src_h, src_mask, s_prev):
        """单步解码（**推理专用**）。

            prev_token : (B,) 或 (B,1) int64   上一步生成的 token
            src_h      : (B,S,Hs)              编码器隐状态序列
            src_mask   : (B,S)
            s_prev     : (B,H)                 上一步隐状态（第一步传 initial_state 的结果）

        返回 (logits (B,V), weights (B,S), s_new (B,H))

        ⚠ 这个方法会**覆盖** emb / attn / cell / out 的前向缓存，
          所以只能用于推理，绝对不能和 backward 混用。
          它就是 teacher forcing 的 `forward` 在 T=1 时的特化版本，
          少了两件推理时不需要的事：保存 cache、算整段序列。
        """
        prev = np.asarray(prev_token, dtype=np.int64).reshape(-1, 1)
        src_h = np.ascontiguousarray(src_h, dtype=np.float64)
        src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
        s_prev = np.ascontiguousarray(s_prev, dtype=np.float64)

        emb_t = self.emb.forward(prev)[:, 0, :]                       # (B,E)
        ctx3, w3 = self.attn.forward(s_prev[:, None, :], src_h, src_h, src_mask)
        ctx = ctx3[:, 0, :]                                          # (B,Hs)
        weights = w3[:, 0, :]                                        # (B,S)

        x_t = np.concatenate([emb_t, ctx], axis=-1)
        s_new = self.cell.forward(x_t, s_prev)

        o_t = self._build_output_input(s_new, ctx, emb_t)
        logits = self.out.forward(o_t)                               # (B,V)
        return logits, weights, s_new

    def __repr__(self) -> str:
        return (f"AttentionDecoder(params={self.num_params()}, "
                f"output_mode={self.output_mode!r}, init_mode={self.init_mode!r}, "
                f"attn={type(self.attn).__name__})")
