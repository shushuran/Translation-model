"""把编码器、解码器、损失函数组装成完整的 Seq2Seq 模型，并提供推理/评估。

    src ──encoder──▶ H_seq, h_final ──decoder──▶ logits ──loss──▶ 标量
                                                    │
                                        argmax ◀────┴────▶ 贪心解码（自回归）

为什么把三者放进一个 Module
-------------------------
`Module.named_parameters()` 是递归的，所以优化器只要拿一次 `model`，
就能同时看到编码器的 embedding/GRU、解码器的注意力/GRU/输出层 ——
不用手工拼前缀，也不会漏掉参数。`zero_grad()` 同理。
"""

from __future__ import annotations

import numpy as np

from .decoder import AttentionDecoder
from .encoder import GRUEncoder
from .losses import CrossEntropyLoss
from .nn import Module

__all__ = ["Seq2Seq", "greedy_decode"]


class Seq2Seq(Module):
    """编码器 + 带注意力的解码器 + 交叉熵损失。"""

    def __init__(self, encoder: GRUEncoder, decoder: AttentionDecoder,
                 loss_fn: CrossEntropyLoss | None = None) -> None:
        super().__init__()
        self.enc = self.register_module("enc", encoder)
        self.dec = self.register_module("dec", decoder)
        self.loss = self.register_module("loss", loss_fn or CrossEntropyLoss())
        self._last: dict | None = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def build(cls, cfg, src_vocab_size: int, tgt_vocab_size: int,
              src_pad_id: int, tgt_pad_id: int, seed: int = 0) -> "Seq2Seq":
        """按 ModelConfig 造一个模型。"""
        enc = GRUEncoder(src_vocab_size, cfg.src_emb_dim, cfg.hidden_dim,
                         init_scale=cfg.init_scale, seed=seed, pad_id=src_pad_id,
                         zero_pad_embedding=True)
        dec = AttentionDecoder(tgt_vocab_size, cfg.tgt_emb_dim, cfg.hidden_dim,
                               cfg.hidden_dim, cfg,
                               output_mode=cfg.decoder_output,
                               init_mode=cfg.decoder_init,
                               init_scale=cfg.init_scale, seed=seed, pad_id=tgt_pad_id)
        return cls(enc, dec)

    # ------------------------------------------------------------------ 前向
    def forward(self, batch, return_weights: bool = False):
        """一个 batch 的完整前向。返回 (loss, logits, weights)。"""
        h_seq, h_final = self.enc.forward(batch.src, batch.src_mask)
        logits, weights = self.dec.forward(batch.tgt_in, h_seq, batch.src_mask, h_final)
        loss = self.loss.forward(logits, batch.tgt_out, batch.tgt_mask)
        self._last = {
            "loss": loss,
            "logits": logits,
            "weights": weights,
            "h_seq": h_seq,
            "h_final": h_final,
            "batch": batch,
        }
        if return_weights:
            return loss, logits, weights
        return loss, logits, weights

    # ------------------------------------------------------------------ 反向
    def backward(self) -> None:
        """从 loss 一路反传到编码器。

        调用前必须先 `model.zero_grad()`（backward 只累加，不清零）。
        """
        if self._last is None:
            raise RuntimeError("调用 backward 之前必须先调用 forward")
        dlogits = self.loss.backward(1.0)
        d = self.dec.backward(dlogits)
        self.enc.backward(d["src_h"], d["src_h_final"])

    # ------------------------------------------------------------------ 便捷属性
    @property
    def last_loss(self) -> float:
        return float(self._last["loss"]) if self._last else float("nan")

    @property
    def last_accuracy(self) -> float:
        return float(self.loss.last_accuracy)

    def encode(self, src, src_mask):
        return self.enc.forward(src, src_mask)

    def __repr__(self) -> str:
        return (f"Seq2Seq(params={self.num_params():,}, "
                f"enc={self.enc.num_params():,}, dec={self.dec.num_params():,}, "
                f"attn={type(self.dec.attn).__name__})")


# --------------------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------------------
def greedy_decode(
    model: Seq2Seq,
    src,
    src_mask,
    max_len: int,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    collect_weights: bool = True,
):
    """自回归贪心解码：每步取概率最大的词，把**自己生成的词**当下一步输入。

    这是推理与训练最本质的差别：
        训练（teacher forcing）：第 t 步的输入永远是**真实的** y_{t-1}
        推理（本函数）          ：第 t 步的输入是**模型自己生成的** ŷ_{t-1}

    一旦某一步生成错了，后面的输入就偏离训练分布，误差会累积 ——
    这就是 **exposure bias**。

    返回 (sequences (B, T'), weights (B, T', S))；序列里含 `<eos>`，其后补 `<pad>`。
    """
    src = np.asarray(src, dtype=np.int64)
    src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError(f"src 必须是二维 (B,S)，收到 {src.shape}")

    B, S = src.shape
    if max_len <= 0:
        raise ValueError(f"max_len 必须为正，收到 {max_len}")

    h_seq, h_final = model.enc.forward(src, src_mask)
    s, _ = model.dec.initial_state(h_final)

    prev = np.full((B,), bos_id, dtype=np.int64)
    out = np.full((B, max_len), pad_id, dtype=np.int64)
    weights = np.zeros((B, max_len, S), dtype=np.float64) if collect_weights else None
    finished = np.zeros((B,), dtype=bool)
    lengths = np.zeros((B,), dtype=np.int64)

    for t in range(max_len):
        logits, w, s_new = model.dec.decode_step(prev, h_seq, src_mask, s)
        nxt = logits.argmax(axis=-1).astype(np.int64)

        # 已经结束的样本：写出 <pad>，状态冻结（用 where 而不是 if，保持批量并行）
        out[:, t] = np.where(finished, pad_id, nxt)
        if collect_weights:
            weights[:, t, :] = w
        s = np.where(finished[:, None], s, s_new)

        newly = (~finished) & (nxt == eos_id)
        lengths = np.where(newly, t + 1, lengths)
        finished = finished | newly
        prev = np.where(finished, eos_id, nxt)

        if finished.all():
            out = out[:, : t + 1]
            if collect_weights:
                weights = weights[:, : t + 1]
            break

    lengths = np.where(lengths == 0, out.shape[1], lengths)
    return out, weights, lengths
