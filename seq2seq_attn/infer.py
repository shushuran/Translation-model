"""推理：Beam Search，以及用来验证它正确性的暴力枚举参考实现。

    greedy     每步只保留 1 条最优前缀
    beam       每步保留 beam_size 条最优前缀，最后按「长度归一化的对数概率」选一条

评分用**对数概率之和**而不是概率之积
--------------------------------
    P(y) = Π p(y_t)   —— 序列一长，乘积就下溢成 0（10 个 0.1 相乘就是 1e-10）
    log P(y) = Σ log p(y_t)   —— 变成求和，数值稳定，而且单调等价

长度惩罚（GNMT 式，Wu et al. 2016）
---------------------------------
    score = log P(y) / ((5 + |y|) / 6)^α

**为什么需要它**：纯对数概率之和**天然偏好短句** —— 每多生成一个词就多一个
负项（log p ≤ 0）。模型可能因此倾向于"赶紧输出 `<eos>`"。
α = 0 时无惩罚；α 越大越鼓励长句；α = 0.6 是 GNMT 及后来多数系统的常用值。

注意分母是 `((5+|y|)/6)^α` 而不是 `|y|^α`：这样**乘性因子**比加性惩罚更平滑，
且 α 对短句和长句的作用连续（GNMT 原文的 `lp` 定义）。

⚠ 什么时候 Beam Search 没用
--------------------------
Beam 是在一个**受限宽度**的搜索里找更高分的序列。如果
  1. 贪心本身已经 100% 正确（搜索空间里没有更好的解），或者
  2. 模型的每步概率严重失准（分数高的序列并不是对的序列），
那么 Beam 不会带来提升 —— 甚至可能更差（它优化的是模型分数，不是准确率）。
本项目充分训练后的 checkpoint 就属于情况 1。
"""

from __future__ import annotations

import numpy as np

from . import data as D
from .model import Seq2Seq, greedy_decode

__all__ = [
    "greedy_decode",
    "beam_search",
    "brute_force_decode",
    "length_penalized_score",
    "decode_dataset",
    "evaluate_decoding",
    "compare_decoding",
]


# --------------------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------------------
def length_penalty_factor(length: int, alpha: float) -> float:
    """GNMT 长度惩罚的分母 `((5 + |y|) / 6)^α`。"""
    if alpha <= 0.0:
        return 1.0
    return float(((5.0 + max(length, 1)) / 6.0) ** alpha)


def length_penalized_score(log_prob: float, length: int, alpha: float) -> float:
    """长度归一化后的分数。α = 0 时就是原始对数概率。"""
    return float(log_prob) / length_penalty_factor(length, alpha)


def _log_softmax_row(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    shifted = x - m
    return shifted - float(np.log(np.exp(shifted).sum()))


# --------------------------------------------------------------------------------------
# Beam Search
# --------------------------------------------------------------------------------------
def beam_search(
    model: Seq2Seq,
    src,
    src_mask,
    max_len: int,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    collect_weights: bool = False,
):
    """逐样本 Beam Search（beam_size = 1 时退化成贪心）。

    每步的流程
    ----------
        1. 对**每条** beam 的当前前缀，算出下一个词的对数概率分布
        2. 取概率最高的 beam_size 个候选词，展开出 beam_size² 个候选
        3. 其中生成 `<eos>` 的移入「已完成」池；其余按分数排序，保留前 beam_size 条
        4. 已完成池里凑够 beam_size 条就提前停止

    复杂度：每个样本 `max_len × beam_size` 次单步解码（贪心是 `max_len` 次）。

    返回 (sequences (B,T'), scores (B,), weights (B,T',S) 或 None)
    """
    src = np.asarray(src, dtype=np.int64)
    src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError(f"src 必须是二维 (B,S)，收到 {src.shape}")
    if beam_size < 1:
        raise ValueError(f"beam_size 必须 ≥ 1，收到 {beam_size}")
    if max_len <= 0:
        raise ValueError(f"max_len 必须为正，收到 {max_len}")

    B, S = src.shape
    h_seq, h_final = model.enc.forward(src, src_mask)
    V = model.dec.vocab_size

    out = np.full((B, max_len), pad_id, dtype=np.int64)
    scores = np.full((B,), -np.inf, dtype=np.float64)
    weights = np.zeros((B, max_len, S), dtype=np.float64) if collect_weights else None
    lengths = np.zeros((B,), dtype=np.int64)

    for b in range(B):
        one_h = h_seq[b : b + 1]
        one_mask = src_mask[b : b + 1]
        s0, _ = model.dec.initial_state(h_final[b : b + 1])

        # 每条 beam = (已生成 token 列表（不含 <bos>）, 累计对数概率, 当前隐状态)
        beams: list[tuple[list[int], float, np.ndarray]] = [([], 0.0, s0)]
        done: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            if not beams:
                break
            candidates: list[tuple[list[int], float, np.ndarray]] = []
            for tokens, logp, state in beams:
                prev = np.array([tokens[-1] if tokens else bos_id], dtype=np.int64)
                logits, _, s_new = model.dec.decode_step(prev, one_h, one_mask, state)
                log_probs = _log_softmax_row(logits[0])
                # 取概率最高的 beam_size 个词（argpartition 是 O(V)，比全排序快）
                k = min(beam_size, V)
                top = np.argpartition(-log_probs, k - 1)[:k] if k < V else np.arange(V)
                for tok in top:
                    tok = int(tok)
                    candidates.append((tokens + [tok], logp + float(log_probs[tok]), s_new))

            # 按长度归一化分数排序（长度 = 已生成 token 数）
            candidates.sort(
                key=lambda c: length_penalized_score(c[1], max(len(c[0]), 1), length_penalty),
                reverse=True,
            )

            next_beams: list[tuple[list[int], float, np.ndarray]] = []
            for tokens, logp, state in candidates:
                if tokens[-1] == eos_id:
                    done.append((tokens, logp))
                elif len(next_beams) < beam_size:
                    next_beams.append((tokens, logp, state))
            beams = next_beams

        # ⚠ 最终的候选池必须同时包含「已完成的假设」和「跑满 max_len 仍未收尾的 beam」。
        #
        # 这里曾经写过一个看起来合理的提前停止：
        #     if len(done) >= beam_size: break
        # 它是**错的**，而且错得很隐蔽：`done` 里凑够了 beam_size 条，
        # 不代表存活 beam 里没有更好的候选。
        #
        # 实测后果（3000 步的 checkpoint，beam_size=3）：
        #     贪心整句准确率 100%  ->  Beam 掉到 60%
        # 调试后发现：正确前缀当时还在 `beams` 里，下一步就能以 log p ≈ 0 收尾
        # （总分 ≈ 0），但 `done` 里已经攒了 3 条 log p ≈ -10 的「提前输出 <eos>」
        # 假设，于是循环被提前掐断，把一个分数 -6.35 的差答案当成了最优解。
        #
        # 为什么「凑够 beam_size 条就停」在标准实现里看起来没事？
        # 因为常见的实现里长度惩罚、打分口径各不相同，掩盖了这个问题。
        # 严谨的停止条件需要「已完成的最优分数 > 存活 beam 分数的上界」，
        # 而长度惩罚让这个上界很难算（序列变长时分数不一定变差）。
        # 所以这里选择**不提前停**：跑到 max_len 或所有 beam 都收尾为止。
        # 代价最多是 max_len × beam_size 次单步解码，对本项目完全可以接受。
        pool = done + [(tokens, logp) for tokens, logp, _ in beams]
        if not pool:
            continue
        best_tokens, best_logp = max(
            pool, key=lambda c: length_penalized_score(c[1], max(len(c[0]), 1), length_penalty)
        )
        best_tokens = best_tokens[:max_len]
        out[b, : len(best_tokens)] = best_tokens
        scores[b] = length_penalized_score(best_logp, max(len(best_tokens), 1), length_penalty)
        lengths[b] = len(best_tokens)

    return out, scores, weights, lengths


# --------------------------------------------------------------------------------------
# 暴力枚举（验证用）
# --------------------------------------------------------------------------------------
def brute_force_decode(
    model: Seq2Seq,
    src,
    src_mask,
    max_len: int,
    length_penalty: float = 0.6,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 0,
    max_nodes: int = 2_000_000,
):
    """**枚举全部序列**，返回分数最高的那一条。只适用于极小的词表 / 长度。

    用途：给 Beam Search 提供一个绝对正确的参照物。
        beam_size ≥ 序列总数时，Beam Search 的结果必须与暴力枚举**完全一致**。

    因为模型是自回归的，可以用 DFS 共享前缀，节点数 = 满 V 叉树的节点数
        Σ_{k=0}^{max_len} V^k
    V = 27、max_len = 10 时这是 10^14 量级，所以函数开头会检查并报错，
    防止误用在大问题上。
    """
    src = np.asarray(src, dtype=np.int64)
    src_mask = np.ascontiguousarray(src_mask, dtype=np.float64)
    B, S = src.shape
    V = model.dec.vocab_size

    upper = sum(V ** k for k in range(max_len + 1))
    if upper > max_nodes:
        raise ValueError(
            f"暴力枚举需要 {upper:.3e} 个节点，超过上限 {max_nodes}。"
            f"请把词表/长度调小（当前 V={V}, max_len={max_len}）。"
        )

    h_seq, h_final = model.enc.forward(src, src_mask)
    out = np.full((B, max_len), pad_id, dtype=np.int64)
    scores = np.full((B,), -np.inf, dtype=np.float64)

    for b in range(B):
        one_h = h_seq[b : b + 1]
        one_mask = src_mask[b : b + 1]
        s0, _ = model.dec.initial_state(h_final[b : b + 1])

        best_tokens: list[int] = []
        best_score = -np.inf
        stack: list[tuple[list[int], float, np.ndarray]] = [([], 0.0, s0)]

        while stack:
            tokens, logp, state = stack.pop()
            finished = bool(tokens) and tokens[-1] == eos_id
            if finished or len(tokens) >= max_len:
                sc = length_penalized_score(logp, max(len(tokens), 1), length_penalty)
                if sc > best_score:
                    best_score, best_tokens = sc, list(tokens)
                continue
            prev = np.array([tokens[-1] if tokens else bos_id], dtype=np.int64)
            logits, _, s_new = model.dec.decode_step(prev, one_h, one_mask, state)
            log_probs = _log_softmax_row(logits[0])
            for tok in range(V - 1, -1, -1):      # 倒序压栈，pop 时从 0 开始
                stack.append((tokens + [tok], logp + float(log_probs[tok]), s_new))

        best_tokens = best_tokens[:max_len]
        out[b, : len(best_tokens)] = best_tokens
        scores[b] = best_score
    return out, scores


# --------------------------------------------------------------------------------------
# 数据集级评估
# --------------------------------------------------------------------------------------
def decode_dataset(
    model: Seq2Seq,
    dataset: D.DateDataset,
    max_len: int,
    method: str = "greedy",
    beam_size: int = 3,
    length_penalty: float = 0.6,
    batch_size: int = 64,
    max_batches: int | None = None,
    collect_weights: bool = False,
):
    """在数据集上批量解码，返回 (hypotheses, references) 两组 id 列表。

    hypotheses / references 都已去掉 `<pad>`，但保留 `<eos>`（这样"忘了停止"
    和"提前停止"都会被算成错误）。
    """
    batches = D.make_batches(dataset, batch_size, shuffle=False)
    if max_batches is not None:
        batches = batches[:max_batches]

    tv = dataset.tgt_vocab
    hyps: list[list[int]] = []
    refs: list[list[int]] = []
    all_weights: list[np.ndarray] = []

    for batch in batches:
        if method == "greedy":
            seqs, ws, _ = greedy_decode(
                model, batch.src, batch.src_mask, max_len,
                bos_id=tv.bos_id, eos_id=tv.eos_id, pad_id=tv.pad_id,
                collect_weights=collect_weights,
            )
        elif method == "beam":
            seqs, _, ws, _ = beam_search(
                model, batch.src, batch.src_mask, max_len,
                beam_size=beam_size, length_penalty=length_penalty,
                bos_id=tv.bos_id, eos_id=tv.eos_id, pad_id=tv.pad_id,
                collect_weights=collect_weights,
            )
        else:
            raise ValueError(f"未知 method: {method!r}（应为 'greedy' 或 'beam'）")

        for i in range(batch.batch_size):
            hyps.append([int(v) for v in seqs[i] if int(v) != tv.pad_id])
            refs.append([int(v) for v in batch.tgt_out[i] if int(v) != tv.pad_id])
            if collect_weights and ws is not None:
                all_weights.append(ws[i].copy())

    if collect_weights:
        return hyps, refs, all_weights
    return hyps, refs


def score_hypotheses(hyps, refs, pad_id: int = 0) -> dict:
    """算 token 准确率、整句准确率、平均输出长度。

    对齐方式：逐位置比较到 `min(len(ref), len(hyp))`，短的一方缺的位置算错。
    这样"提前停止"和"忘了停止"都会被惩罚。
    """
    n_tok = n_tok_correct = 0
    n_seq = n_seq_correct = 0
    total_len = 0
    for h, r in zip(hyps, refs):
        h = [int(x) for x in h if int(x) != pad_id]
        r = [int(x) for x in r if int(x) != pad_id]
        m = min(len(h), len(r))
        n_tok += len(r)
        n_tok_correct += sum(1 for j in range(m) if h[j] == r[j])
        n_seq += 1
        n_seq_correct += int(h == r)
        total_len += len(h)
    return {
        "token_acc": n_tok_correct / max(n_tok, 1),
        "seq_acc": n_seq_correct / max(n_seq, 1),
        "mean_len": total_len / max(n_seq, 1),
        "n_seq": n_seq,
    }


def evaluate_decoding(
    model: Seq2Seq,
    dataset: D.DateDataset,
    max_len: int,
    method: str = "greedy",
    beam_size: int = 3,
    length_penalty: float = 0.6,
    batch_size: int = 64,
    max_batches: int | None = None,
) -> dict:
    """在数据集上评估一种解码方式。"""
    hyps, refs = decode_dataset(
        model, dataset, max_len, method=method, beam_size=beam_size,
        length_penalty=length_penalty, batch_size=batch_size, max_batches=max_batches,
    )
    stats = score_hypotheses(hyps, refs, pad_id=dataset.tgt_vocab.pad_id)
    stats["method"] = method if method == "greedy" else f"beam{beam_size}"
    stats["length_penalty"] = length_penalty
    return stats


def compare_decoding(
    model: Seq2Seq,
    dataset: D.DateDataset,
    max_len: int,
    beam_sizes=(1, 3, 5, 10),
    length_penalty: float = 0.6,
    batch_size: int = 64,
    max_batches: int | None = None,
    include_greedy: bool = True,
) -> list[dict]:
    """对比贪心与不同 beam_size 的效果。"""
    rows: list[dict] = []
    if include_greedy:
        rows.append(evaluate_decoding(model, dataset, max_len, method="greedy",
                                      batch_size=batch_size, max_batches=max_batches))
    for bs in beam_sizes:
        rows.append(evaluate_decoding(model, dataset, max_len, method="beam", beam_size=bs,
                                      length_penalty=length_penalty,
                                      batch_size=batch_size, max_batches=max_batches))
    return rows
