"""Beam Search / 贪心解码 / 暴力枚举参照 的断言测试。

运行：
    python tests/test_infer.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import numpy as np

from seq2seq_attn import data as D
from seq2seq_attn import train as T
from seq2seq_attn.config import CONFIG
from seq2seq_attn.decoder import AttentionDecoder
from seq2seq_attn.encoder import GRUEncoder
from seq2seq_attn.infer import (
    beam_search,
    brute_force_decode,
    decode_dataset,
    evaluate_decoding,
    length_penalized_score,
    length_penalty_factor,
    score_hypotheses,
)
from seq2seq_attn.model import Seq2Seq, greedy_decode
from seq2seq_attn.nn import log_softmax

PASSED = 0
CFG = CONFIG.model


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        raise AssertionError(name)
    PASSED += 1


def close(a, b, tol=1e-12) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=0))


def tiny_model(vocab=4, seed=0, src_vocab=6):
    return Seq2Seq(
        GRUEncoder(src_vocab, 4, 5, seed=seed),
        AttentionDecoder(vocab, 4, 5, 5, CFG, seed=seed),
    )


# --------------------------------------------------------------------------------------
# 脚本化的假模型：按预设「下一个词的 logits」返回，用于精确测试 beam 的剪枝与终止逻辑
# --------------------------------------------------------------------------------------
class _ScriptedEncoder:
    def forward(self, src, src_mask):
        return (np.zeros((src.shape[0], src.shape[1], 1)),
                np.zeros((src.shape[0], 1)))


class _ScriptedDecoder:
    def __init__(self, table, vocab_size, pad_id=0):
        self._table = table
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def initial_state(self, h_final):
        return np.zeros((h_final.shape[0], 1)), None

    def decode_step(self, prev_token, src_h, src_mask, s_prev):
        tok = int(np.asarray(prev_token).reshape(-1)[0])
        logits = np.asarray(self._table[tok], dtype=np.float64)[None, :]
        weights = np.full((1, src_h.shape[1]), 1.0 / src_h.shape[1])
        return logits, weights, s_prev


class _ScriptedModel:
    """只提供 beam_search / greedy_decode 需要的最小接口。"""

    def __init__(self, table, vocab_size, pad_id=0):
        self.enc = _ScriptedEncoder()
        self.dec = _ScriptedDecoder(table, vocab_size, pad_id)


# --------------------------------------------------------------------------------------
def test_length_penalty() -> None:
    print("\n== 长度惩罚 ==")
    check("α=0 时因子为 1", length_penalty_factor(1, 0.0) == 1.0 and
          length_penalty_factor(50, 0.0) == 1.0)
    check("α>0 时因子随长度单调递增",
          all(length_penalty_factor(n, 0.6) < length_penalty_factor(n + 1, 0.6)
              for n in range(1, 30)))
    check("|y|=1 时因子为 1（GNMT 的归一化点）",
          abs(length_penalty_factor(1, 0.6) - 1.0) < 1e-12,
          f"{length_penalty_factor(1, 0.6):.10f}")
    check("α=0 时分数就是原始对数概率",
          length_penalized_score(-10.0, 5, 0.0) == -10.0)
    check("α=0 时长短句分数相同（没有长度偏好）",
          length_penalized_score(-10.0, 1, 0.0) == length_penalized_score(-10.0, 30, 0.0))
    # 注意方向：长度惩罚**不降低**短句分数，而是**抬升**长句分数
    #   |y|=1 时因子恰为 1，所以短句分数不变
    #   代入数字：score(-10, 1) = -10/1 = -10
    #             score(-10, 2) = -10/((7/6)^0.6)   = -9.359
    #             score(-10,10) = -10/((15/6)^0.6)  = -5.790
    check("α>0 时相同对数概率下长句分数更高（补偿长度）",
          length_penalized_score(-10.0, 2, 0.6) > length_penalized_score(-10.0, 1, 0.6) and
          length_penalized_score(-10.0, 10, 0.6) > length_penalized_score(-10.0, 2, 0.6))
    check("α>0 相对 α=0 只抬升长句、不动短句",
          length_penalized_score(-10.0, 1, 0.6) == length_penalized_score(-10.0, 1, 0.0) and
          length_penalized_score(-10.0, 10, 0.6) > length_penalized_score(-10.0, 10, 0.0))
    # α 越大，长句得到的补偿越多
    check("α 越大对长句越有利",
          length_penalized_score(-10.0, 20, 1.5) > length_penalized_score(-10.0, 20, 0.6))


def test_score_hypotheses() -> None:
    print("\n== 序列打分 ==")
    hyps = [[1, 2, 3], [1, 9, 3], [1, 2], [1, 2, 3, 4]]
    refs = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    s = score_hypotheses(hyps, refs)
    check("整句准确率 = 1/4", abs(s["seq_acc"] - 0.25) < 1e-12, f"{s['seq_acc']:.4f}")
    # token：正确数 = 3（第0条）+ 2（第1条）+ 2（第2条，长度不足）+ 3（第3条，多出的不算）
    check("token 准确率 = 10/12", abs(s["token_acc"] - 10 / 12) < 1e-12, f"{s['token_acc']:.4f}")
    check("n_seq 正确", s["n_seq"] == 4)
    check("平均长度正确", abs(s["mean_len"] - (3 + 3 + 2 + 4) / 4) < 1e-12)

    check("提前停止被算成错误",
          score_hypotheses([[1, 2]], [[1, 2, 3]])["seq_acc"] == 0.0)
    check("忘了停止被算成错误",
          score_hypotheses([[1, 2, 3, 4]], [[1, 2, 3]])["seq_acc"] == 0.0)
    check("pad 被忽略",
          score_hypotheses([[1, 2, 0, 0]], [[1, 2]])["seq_acc"] == 1.0)
    check("空输入不崩", score_hypotheses([], [])["n_seq"] == 0)


def test_beam_equals_greedy_at_one() -> None:
    print("\n== beam_size=1 必须退化成贪心 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    tv = train.tgt_vocab
    kw = dict(bos_id=tv.bos_id, eos_id=tv.eos_id, pad_id=tv.pad_id)

    batch = D.collate(test, list(range(8)))
    for lp in (0.0, 0.6, 1.5):
        g, _, _ = greedy_decode(model, batch.src, batch.src_mask, 12, **kw)
        b, _, _, _ = beam_search(model, batch.src, batch.src_mask, 12,
                                 beam_size=1, length_penalty=lp, **kw)
        check(f"α={lp}: beam1 == greedy（逐位）", bool(np.array_equal(g, b)))


def test_beam_matches_brute_force() -> None:
    print("\n== beam_size 足够大时必须等于暴力枚举 ==")
    src = np.array([[1, 2, 3, 4, 0], [5, 4, 3, 2, 1]])
    mask = np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    V, max_len = 4, 3
    # 序列总数上界 = Σ V^k，k=0..3 = 1+4+16+64 = 85

    for seed in (0, 1, 2):
        model = tiny_model(vocab=V, seed=seed)
        for lp in (0.0, 0.6):
            bf, bf_scores = brute_force_decode(model, src, mask, max_len, length_penalty=lp)
            # beam_size = V 时每步保留全部候选，等价于穷举
            bs, bs_scores, _, _ = beam_search(model, src, mask, max_len,
                                              beam_size=V, length_penalty=lp)
            check(f"seed={seed} α={lp}: beam{V} == 暴力枚举",
                  bool(np.array_equal(bf, bs)), f"{bf.tolist()} vs {bs.tolist()}")
            check(f"seed={seed} α={lp}: 分数一致",
                  close(bf_scores, bs_scores, tol=1e-11),
                  f"{bf_scores.round(6).tolist()} vs {bs_scores.round(6).tolist()}")

    # 反向验证：暴力枚举确实能优于贪心（否则这个测试没有区分度）
    model = tiny_model(vocab=V, seed=0)
    bf, bf_s = brute_force_decode(model, src, mask, max_len, length_penalty=0.0)
    g, _, _ = greedy_decode(model, src, mask, max_len,
                            bos_id=1, eos_id=2, pad_id=0)
    g_scores = _score_sequences(model, src, mask, g, length_penalty=0.0)
    check("暴力枚举确实能优于贪心（测试有区分度）",
          bool(np.any(bf_s > g_scores + 1e-9)),
          f"brute {bf_s.round(4).tolist()} vs greedy {g_scores.round(4).tolist()}")


def _score_sequences(model, src, mask, seqs, length_penalty=0.0) -> np.ndarray:
    """用**教师强制前向**重算给定序列的对数概率（独立于 beam 内部累加）。"""
    tv_bos = model.dec.pad_id  # 占位，下面显式传
    return np.array([
        _score_one(model, src[i:i + 1], mask[i:i + 1], seqs[i], length_penalty)
        for i in range(len(seqs))
    ])


def _score_one(model, src, mask, seq, length_penalty) -> float:
    tokens = [int(t) for t in seq if int(t) != 0]
    if not tokens:
        return 0.0
    h_seq, h_final = model.enc.forward(src, mask)
    tgt_in = np.array([[1] + tokens[:-1]], dtype=np.int64)     # bos + y[:-1]
    logits, _ = model.dec.forward(tgt_in, h_seq, mask, h_final)
    logp = log_softmax(logits[0])
    total = sum(float(logp[t, tokens[t]]) for t in range(len(tokens)))
    return length_penalized_score(total, len(tokens), length_penalty)


def test_beam_score_is_consistent() -> None:
    print("\n== Beam 返回的分数必须与独立重算一致 ==")
    print("  （用教师强制前向重算 Σ log p，验证内部累加没有错）")
    train, dev, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    batch = D.collate(test, list(range(5)))

    for lp in (0.0, 0.6, 1.5):
        seqs, scores, _, lengths = beam_search(
            model, batch.src, batch.src_mask, 12, beam_size=4, length_penalty=lp,
            bos_id=train.tgt_vocab.bos_id, eos_id=train.tgt_vocab.eos_id,
            pad_id=train.tgt_vocab.pad_id,
        )
        recomputed = _score_sequences(model, batch.src, batch.src_mask, seqs,
                                      length_penalty=lp)
        check(f"α={lp}: 分数与独立重算一致", close(scores, recomputed, tol=1e-10),
              f"max差 {np.abs(scores - recomputed).max():.2e}")
        check(f"α={lp}: 长度 = 非 pad token 数",
              bool(np.all(lengths == (seqs != train.tgt_vocab.pad_id).sum(axis=1))),
              f"{lengths.tolist()} vs {(seqs != train.tgt_vocab.pad_id).sum(axis=1).tolist()}")


def test_beam_output_structure() -> None:
    print("\n== 输出结构 ==")
    train, _, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    batch = D.collate(test, list(range(6)))
    pad_id, eos_id, bos_id = (train.tgt_vocab.pad_id, train.tgt_vocab.eos_id,
                              train.tgt_vocab.bos_id)

    seqs, scores, ws, lengths = beam_search(
        model, batch.src, batch.src_mask, 12, beam_size=3,
        bos_id=bos_id, eos_id=eos_id, pad_id=pad_id, collect_weights=True)

    check("序列形状 (B,T')", seqs.shape == (6, 12), str(seqs.shape))
    check("分数有限", bool(np.all(np.isfinite(scores))))
    check("输出无 nan", bool(np.all(seqs >= 0)))
    check("权重形状对齐", ws.shape == (6, 12, batch.src.shape[1]), str(ws.shape))

    ok = True
    for b in range(6):
        pos = np.where(seqs[b] == eos_id)[0]
        if pos.size and np.any(seqs[b, pos[0] + 1:] != pad_id):
            ok = False
        # <eos> 之前不能出现 <pad>
        if pos.size and np.any(seqs[b, : pos[0]] == pad_id):
            ok = False
    check("<eos> 之后全是 <pad>，之前没有 <pad>", ok)

    # 确定性
    s2, sc2, _, l2 = beam_search(model, batch.src, batch.src_mask, 12, beam_size=3,
                                 bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    check("两次结果完全相同", bool(np.array_equal(seqs, s2)) and close(scores, sc2))

    # 逐样本 == 批量
    one, one_sc, _, _ = beam_search(model, batch.src[2:3], batch.src_mask[2:3], 12,
                                    beam_size=3, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)
    check("逐样本结果与批量一致",
          bool(np.array_equal(one[0], seqs[2])) and abs(one_sc[0] - scores[2]) < 1e-12)

    # 更多 beam 不应让「已探索的最优」变差 —— 用暴力枚举在小模型上验证过，
    # 这里只检查权重收集开关
    _, _, ws_off, _ = beam_search(model, batch.src, batch.src_mask, 12, beam_size=3,
                                  bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                                  collect_weights=False)
    check("collect_weights=False 时不返回权重", ws_off is None)


def test_premature_stop_regression() -> None:
    print("\n== 回归测试：不能因为「已完成假设攒够了」就提前停止 ==")
    print("  （这是本项目第 6 步真的踩到的 bug：")
    print("    `if len(done) >= beam_size: break` 会把存活 beam 里更好的候选丢掉。")
    print("    实测后果：3000 步的 checkpoint 上贪心 100% → Beam3 掉到 60%。）\n")

    PAD, BOS, EOS, A_, B_, C_, D_ = 0, 1, 2, 3, 4, 5, 6
    V = 7
    NEG = -10.0
    # 从 <bos> 出发：A 0.4 / B 0.3 / C 0.2（取 top-3 时全进 beam）
    # 从 A/B/C 出发：各自有很小的概率输出 <eos>，更大的概率输出 D
    # 从 D 出发：几乎必然输出 <eos>
    # 于是第 2 步时 done 就会攒到 3 条「提前收尾」的假设，
    # 但真正最优的序列是 A → D → <eos>，它要到第 3 步才收尾。
    table = {
        BOS: [NEG, NEG, NEG, 1.0, 0.7, 0.4, NEG],
        A_: [NEG, NEG, 0.0, NEG, NEG, NEG, 3.0],
        B_: [NEG, NEG, 0.0, NEG, NEG, NEG, 2.9],
        C_: [NEG, NEG, 0.0, NEG, NEG, NEG, 2.8],
        D_: [NEG, NEG, 5.0, NEG, NEG, NEG, NEG],
    }
    model = _ScriptedModel(table, V, pad_id=PAD)
    src = np.array([[1, 2, 3]])
    mask = np.ones((1, 3))

    def lp(tok, nxt):
        row = np.asarray(table[tok], dtype=float)
        m = row.max()
        return float(row[nxt] - m - np.log(np.exp(row - m).sum()))

    s_A = lp(BOS, A_)
    s_AD = s_A + lp(A_, D_)
    s_ADeos = s_AD + lp(D_, EOS)
    s_Aeos = s_A + lp(A_, EOS)
    check("构造的脚本里「晚收尾」确实更优", s_ADeos > s_Aeos + 1.0,
          f"{s_ADeos:.4f} vs {s_Aeos:.4f}")

    seqs, scores, _, lengths = beam_search(
        model, src, mask, 3, beam_size=3, length_penalty=0.0,
        bos_id=BOS, eos_id=EOS, pad_id=PAD)
    tokens = [int(t) for t in seqs[0] if int(t) != PAD]
    check("返回的是「晚收尾」的最优序列", tokens == [A_, D_, EOS], str(tokens))
    check("分数等于该序列的对数概率", abs(float(scores[0]) - s_ADeos) < 1e-12,
          f"{float(scores[0]):.6f} vs {s_ADeos:.6f}")
    check("长度正确", int(lengths[0]) == 3, str(int(lengths[0])))

    # 贪心也应该给出同一条（这个脚本里贪心恰好最优）
    g, _, _ = greedy_decode(model, src, mask, 3, bos_id=BOS, eos_id=EOS, pad_id=PAD)
    check("贪心给出同一条", [int(t) for t in g[0] if int(t) != PAD] == [A_, D_, EOS],
          str([int(t) for t in g[0] if int(t) != PAD]))


def test_beam_size_monotone_on_tiny() -> None:
    print("\n== 小模型上：beam 越大分数不会变差 ==")
    src = np.array([[1, 2, 3, 4, 0], [5, 4, 3, 2, 1]])
    mask = np.array([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    model = tiny_model(vocab=4, seed=3)
    prev = None
    for K in (1, 2, 3, 4):
        _, sc, _, _ = beam_search(model, src, mask, 3, beam_size=K, length_penalty=0.0)
        if prev is not None:
            check(f"beam{K} 分数 ≥ beam{K - 1}（逐样本）",
                  bool(np.all(sc >= prev - 1e-9)),
                  f"{prev.round(4).tolist()} -> {sc.round(4).tolist()}")
        prev = sc


def test_errors() -> None:
    print("\n== 错误输入 ==")
    train, _, _ = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    src = np.array([[1, 2, 3]])
    mask = np.ones((1, 3))
    for label, kwargs in [("beam_size=0", dict(beam_size=0)),
                          ("beam_size=-1", dict(beam_size=-1)),
                          ("max_len=0", dict(max_len=0))]:
        args = dict(beam_size=3, max_len=3)
        args.update(kwargs)
        try:
            beam_search(model, src, mask, **args)
            check(f"拒绝：{label}", False)
        except ValueError:
            check(f"拒绝：{label}", True)

    try:
        brute_force_decode(model, src, mask, 10)
        check("暴力枚举在大搜索空间上拒绝执行", False)
    except ValueError as exc:
        check("暴力枚举在大搜索空间上拒绝执行", "节点" in str(exc))

    try:
        decode_dataset(model, train.subset(4), 5, method="nope")
        check("拒绝未知 method", False)
    except ValueError:
        check("拒绝未知 method", True)


def test_dataset_level() -> None:
    print("\n== 数据集级解码与评估 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    small = test.subset(24)

    hyps, refs = decode_dataset(model, small, 12, method="greedy")
    check("返回条数与样本数一致", len(hyps) == 24 and len(refs) == 24)
    check("参考序列不含 pad", all(0 not in r for r in refs))
    check("假设序列不含 pad", all(0 not in h for h in hyps))
    check("参考序列以 <eos> 结尾",
          all(r[-1] == train.tgt_vocab.eos_id for r in refs), str(refs[0]))

    hyps_b, refs_b = decode_dataset(model, small, 12, method="beam", beam_size=3)
    check("beam 与 greedy 的参考序列相同", refs == refs_b)

    ev_g = evaluate_decoding(model, small, 12, method="greedy")
    ev_b = evaluate_decoding(model, small, 12, method="beam", beam_size=3)
    for name, ev in (("greedy", ev_g), ("beam", ev_b)):
        check(f"{name} 返回完整字段",
              {"token_acc", "seq_acc", "mean_len", "n_seq", "method"} <= set(ev), str(set(ev)))
        check(f"{name} 的 n_seq 正确", ev["n_seq"] == 24)
        check(f"{name} 的准确率在 [0,1]", 0.0 <= ev["token_acc"] <= 1.0 and
              0.0 <= ev["seq_acc"] <= 1.0)
    check("greedy 的 method 标记", ev_g["method"] == "greedy")
    check("beam 的 method 标记", ev_b["method"] == "beam3")

    # weights 收集
    h, r, w = decode_dataset(model, small, 12, method="beam", beam_size=3,
                             collect_weights=True)
    check("收集权重时返回三个值", len(w) == 24)
    check("权重形状与源长一致", w[0].shape[1] == len(small.examples[0].src_tokens),
          str(w[0].shape))


def test_beam_on_trained_checkpoint_like_perfect() -> None:
    print("\n== 100% 的 checkpoint 上 Beam 不会更差 ==")
    print("  （用未训练的模型代替：只验证两种方法结果结构一致、不会崩）")
    train, _, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CFG)
    small = test.subset(16)
    g = evaluate_decoding(model, small, 12, method="greedy", batch_size=16)
    b = evaluate_decoding(model, small, 12, method="beam", beam_size=3, batch_size=16)
    check("两种方法都能跑完", np.isfinite(g["token_acc"]) and np.isfinite(b["token_acc"]))
    check("beam 的输出长度 ≥ 1", b["mean_len"] >= 1.0, f"{b['mean_len']:.3f}")
    # beam 优化的是模型分数而不是准确率，所以准确率不保证更高 —— 这里只记录
    print(f"        （未训练模型：greedy seq_acc {g['seq_acc']:.3f}，"
          f"beam seq_acc {b['seq_acc']:.3f}；Beam 优化的是模型分数，不是准确率）")


def main() -> None:
    print("=" * 70)
    print(" 推理：Beam Search / 贪心 / 暴力枚举 测试")
    print("=" * 70)
    test_length_penalty()
    test_score_hypotheses()
    test_beam_equals_greedy_at_one()
    test_beam_matches_brute_force()
    test_beam_score_is_consistent()
    test_beam_output_structure()
    test_premature_stop_regression()
    test_beam_size_monotone_on_tiny()
    test_errors()
    test_dataset_level()
    test_beam_on_trained_checkpoint_like_perfect()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
