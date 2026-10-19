"""注意力模块的断言测试（无第三方测试框架依赖）。

运行：
    python tests/test_attention.py

这些断言是第 4 步的安全网：注意力先被证明是对的，拼接解码器时才只可能
是解码器自己的问题。
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

from seq2seq_attn.attention import (
    BahdanauAttention,
    DotProductAttention,
    build_attention,
)
from seq2seq_attn.config import CONFIG
from seq2seq_attn.nn import (
    NEG_INF,
    check_gradient,
    softmax,
    softmax_backward,
    softmax_masked,
    summarize_gradient_report,
)

PASSED = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        raise AssertionError(name)
    PASSED += 1


def close(a, b, tol=1e-12) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=0))


# --------------------------------------------------------------------------------------
def test_softmax() -> None:
    print("\n== softmax ==")
    x = np.array([[1.0, 2.0, 3.0]])
    y = softmax(x)
    check("权重和为 1", close(y.sum(-1), 1.0))
    check("单调性保持", bool(y[0, 0] < y[0, 1] < y[0, 2]))
    check("平移不变", close(softmax(x + 100.0), y))
    check("大数不溢出", bool(np.isfinite(softmax(np.array([[1e4, 1e4 + 1.0]]))).all()))

    # softmax 反向的正确性：与数值梯度对照
    rng = np.random.default_rng(0)
    xs = np.ascontiguousarray(rng.normal(size=(3, 4)))
    g = rng.normal(size=(3, 4))
    w = softmax(xs)
    dx = softmax_backward(w, g)
    eps = 1e-6
    num = np.zeros_like(xs)
    for idx in np.ndindex(xs.shape):
        old = xs[idx]
        xs[idx] = old + eps
        fp = float((softmax(xs) * g).sum())
        xs[idx] = old - eps
        fm = float((softmax(xs) * g).sum())
        xs[idx] = old
        num[idx] = (fp - fm) / (2 * eps)
    check("softmax_backward 与数值梯度一致",
          float(np.max(np.abs(dx - num) / np.maximum(np.abs(dx) + np.abs(num), 1e-12))) < 1e-8)


def test_softmax_masked() -> None:
    print("\n== softmax_masked ==")
    rng = np.random.default_rng(1)
    scores = rng.normal(scale=2.0, size=(3, 7))
    mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0]], dtype=np.float64)

    w = softmax_masked(scores, mask)
    check("输出形状不变", w.shape == scores.shape)
    check("每行和为 1", close(w.sum(-1), 1.0))
    check("mask=0 的位置严格为 0", bool(np.all(w[mask < 0.5] == 0.0)))
    check("mask=1 的位置为正", bool(np.all(w[mask > 0.5] > 0.0)))

    # 与加性 mask 方案在正常量级下逐位一致
    w_a = softmax(scores + (1.0 - mask) * NEG_INF)
    check("与「加 -1e9」方案逐位相同（正常量级）", bool(np.array_equal(w, w_a)))

    # 退化情况：整行被 mask
    alloff = np.zeros((1, 7))
    w_off = softmax_masked(scores[:1], alloff)
    check("整行被 mask 时无 nan", bool(np.isfinite(w_off).all()))
    check("整行被 mask 时权重全 0", bool(np.all(w_off == 0.0)))
    w_off_a = softmax(scores[:1] + (1.0 - alloff) * NEG_INF)
    check("对照：加 -1e9 方案此时静默退化成 softmax(scores)",
          bool(np.allclose(w_off_a, softmax(scores[:1]))))

    # ---- 回归测试：numpy 广播从右对齐的陷阱 ----
    # (B,S) 的 mask 直接和 (B,Tq,S) 的 scores 运算时，Tq==1 会**静默**广播成 (B,B,S)
    sc = rng.normal(size=(4, 1, 5))               # B=4, Tq=1, S=5（解码器的真实形状）
    m2 = np.zeros((4, 5))
    for b in range(4):
        m2[b, : 5 - b] = 1.0
    naive_wrong = np.where(m2 > 0, sc, -np.inf)
    check("确认这个陷阱真的存在（裸 np.where 得到错误形状）", naive_wrong.shape == (4, 4, 5),
          f"实际 {naive_wrong.shape}")

    w2 = softmax_masked(sc, m2)
    check("softmax_masked 修复了形状", w2.shape == (4, 1, 5), f"实际 {w2.shape}")
    ok = all(np.all(w2[b, 0][m2[b] == 0] == 0.0) for b in range(4))
    check("修复后每个样本用自己的 mask", ok)
    ok2 = all(abs(w2[b, 0].sum() - 1.0) < 1e-12 for b in range(4))
    check("修复后每行各自归一化", ok2)

    # 三维 mask 也应支持
    m3 = np.broadcast_to(m2[:, None, :], (4, 1, 5))
    check("三维 mask 与二维 mask 结果一致", bool(np.array_equal(softmax_masked(sc, m3), w2)))

    # 一维 mask (S,) 是允许的：右对齐后恰好落在 key 维上，语义无歧义
    m1 = np.ones(5)
    m1[3:] = 0.0
    w1 = softmax_masked(sc, m1)
    m2_same = np.tile(m1, (4, 1))
    check("一维 mask 等价于所有行相同的二维 mask",
          bool(np.array_equal(w1, softmax_masked(sc, m2_same))))
    check("一维 mask 形状正确", w1.shape == (4, 1, 5), str(w1.shape))


# --------------------------------------------------------------------------------------
def _dims(btype):
    rng = np.random.default_rng(7)
    B, Tq, S, D = 2, 3, 5, 16
    q = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    k = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    v = np.ascontiguousarray(rng.normal(size=(B, S, D + 3)))       # v_dim != k_dim
    mask = np.ones((B, S))
    mask[1, 3:] = 0.0
    return btype, q, k, v, mask


def test_shapes_and_masking() -> None:
    print("\n== 形状与 mask ==")
    for name, factory in [("加性", lambda: BahdanauAttention(16, 16, 19, attn_dim=8, seed=1)),
                          ("点积", lambda: DotProductAttention(16, 16, 19, seed=2))]:
        attn, q, k, v, mask = _dims(factory())
        ctx, w = attn(q, k, v, mask)
        check(f"{name}: context 形状跟随 v_dim", ctx.shape == (2, 3, 19), str(ctx.shape))
        check(f"{name}: weights 形状 (B,Tq,S)", w.shape == (2, 3, 5), str(w.shape))
        check(f"{name}: 每行和为 1", close(w.sum(-1), 1.0))
        check(f"{name}: 填充位置权重为 0",
              bool(np.all(w[np.broadcast_to(mask[:, None, :] < 0.5, w.shape)] == 0.0)))
        check(f"{name}: 样本 0（无 padding）权重全为正", bool(np.all(w[0] > 0)))

    # 形状校验应当大声报错，而不是静默算错
    attn = DotProductAttention(16, 16, 16, seed=3)
    q = np.zeros((2, 1, 16)); k = np.zeros((2, 5, 16)); v = np.zeros((2, 5, 16))
    for label, bad in [("batch 维不一致", (np.zeros((3, 1, 16)), k, v, None)),
                       ("keys/values 长度不一致", (q, k, np.zeros((2, 4, 16)), None)),
                       ("mask 形状错误", (q, k, v, np.ones((2, 4)))),
                       ("mask 维数错误", (q, k, v, np.ones(5)))]:
        try:
            attn(*bad)
            check(f"拒绝：{label}", False)
        except ValueError:
            check(f"拒绝：{label}", True)


def test_param_counts() -> None:
    print("\n== 参数量 ==")
    A = 8
    a = BahdanauAttention(16, 16, 19, attn_dim=A, seed=1)
    expect = 16 * A + 16 * A + A + A
    check("加性参数量 = q_dim·A + k_dim·A + A + A", a.num_params() == expect,
          f"{a.num_params()} vs {expect}")
    check("加性参数集合", set(a.params()) == {"W_q", "W_k", "v", "b"}, str(set(a.params())))

    d0 = DotProductAttention(16, 16, 19, seed=2)
    check("点积（q_dim==k_dim）零参数", d0.num_params() == 0)
    check("点积无参数时 params 为空", len(d0.params()) == 0)

    d1 = DotProductAttention(16, 16, 19, dim=4, seed=3)
    check("点积开启投影后有 W_q / W_k", set(d1.params()) == {"W_q", "W_k"})
    check("点积投影参数量 = q_dim·d + k_dim·d", d1.num_params() == 16 * 4 * 2)

    # q_dim != k_dim 且不投影，必须报错
    try:
        DotProductAttention(16, 32, 32, use_projection=False)
        check("拒绝：q_dim≠k_dim 且不投影", False)
    except ValueError:
        check("拒绝：q_dim≠k_dim 且不投影", True)

    # 工厂函数
    cfg = CONFIG.model
    check("工厂：bahdanau", isinstance(build_attention(cfg, 16, 16, 16), BahdanauAttention))
    import dataclasses
    cfg_dot = dataclasses.replace(cfg, attn_type="dot")
    check("工厂：dot", isinstance(build_attention(cfg_dot, 16, 16, 16), DotProductAttention))
    check("工厂：dot 在 128 维下零参数", build_attention(cfg_dot, 128, 128, 128).num_params() == 0)
    try:
        build_attention(dataclasses.replace(cfg, attn_type="nope"), 16, 16, 16)
        check("工厂：未知类型报错", False)
    except ValueError:
        check("工厂：未知类型报错", True)


def test_naive_equivalence() -> None:
    print("\n== 与朴素三重循环定义逐值比对 ==")
    rng = np.random.default_rng(11)
    B, Tq, S, dq, dk, dv, A = 2, 2, 3, 4, 4, 5, 6
    q = rng.normal(size=(B, Tq, dq))
    k = rng.normal(size=(B, S, dk))
    v = rng.normal(size=(B, S, dv))
    mask = np.ones((B, S)); mask[0, 2] = 0.0

    def naive_ctx(w):
        out = np.zeros((B, Tq, dv))
        for b in range(B):
            for t in range(Tq):
                for x in range(dv):
                    out[b, t, x] = sum(w[b, t, s] * v[b, s, x] for s in range(S))
        return out

    # 加性
    a = BahdanauAttention(dq, dk, dv, attn_dim=A, seed=5)
    ctx, w = a(q, k, v, mask)
    P = a.params()
    sc = np.zeros((B, Tq, S))
    for b in range(B):
        for t in range(Tq):
            for s in range(S):
                sc[b, t, s] = sum(
                    np.tanh(P["b"][i] + sum(q[b, t, d] * P["W_q"][d, i] for d in range(dq))
                            + sum(k[b, s, d] * P["W_k"][d, i] for d in range(dk))) * P["v"][i]
                    for i in range(A))
    sc = np.where(mask[:, None, :] > 0, sc, -np.inf)
    e = np.exp(sc - np.where(np.isfinite(sc.max(-1, keepdims=True)), sc.max(-1, keepdims=True), 0.0))
    w_n = e / np.maximum(e.sum(-1, keepdims=True), 1e-300)
    check("加性 weights 与朴素实现一致", close(w, w_n, tol=1e-12))
    check("加性 context 与朴素实现一致", close(ctx, naive_ctx(w_n), tol=1e-12))
    check("mask 位权重为 0", w[0, 0, 2] == 0.0)

    # 点积
    d = DotProductAttention(dq, dk, dv, seed=6)
    ctx2, w2 = d(q, k, v, mask)
    sc2 = np.zeros((B, Tq, S))
    for b in range(B):
        for t in range(Tq):
            for s in range(S):
                sc2[b, t, s] = sum(q[b, t, x] * k[b, s, x] for x in range(dq)) / np.sqrt(dq)
    sc2 = np.where(mask[:, None, :] > 0, sc2, -np.inf)
    e2 = np.exp(sc2 - np.where(np.isfinite(sc2.max(-1, keepdims=True)), sc2.max(-1, keepdims=True), 0.0))
    w2_n = e2 / np.maximum(e2.sum(-1, keepdims=True), 1e-300)
    check("点积 weights 与朴素实现一致", close(w2, w2_n, tol=1e-12))
    check("点积 context 与朴素实现一致", close(ctx2, naive_ctx(w2_n), tol=1e-12))


def test_hand_computed() -> None:
    print("\n== 手算用例 ==")
    q = np.array([[[1.0, 0.0]]])
    k = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    v = np.array([[[10.0, 0.0], [0.0, 20.0]]])
    attn = DotProductAttention(2, 2, 2, seed=0)
    ctx, w = attn(q, k, v)

    w_expect = np.exp([1 / np.sqrt(2), 0.0])
    w_expect = w_expect / w_expect.sum()
    check("weights 手算一致", close(w.reshape(-1), w_expect))
    ctx_expect = w_expect[0] * np.array([10.0, 0.0]) + w_expect[1] * np.array([0.0, 20.0])
    check("context 手算一致", close(ctx.reshape(-1), ctx_expect))

    # mask 掉第二个 key -> 权重应变成 [1, 0]
    ctx2, w2 = attn(q, k, v, np.array([[1.0, 0.0]]))
    check("单 key 有效时权重为 [1,0]", close(w2.reshape(-1), [1.0, 0.0]))
    check("单 key 有效时 context 等于该 value", close(ctx2.reshape(-1), [10.0, 0.0]))

    # 用 mask 只剩第二个 key -> 权重 [0,1]，context = value1
    ctx3, w3 = attn(q, k, v, np.array([[0.0, 1.0]]))
    check("屏蔽 key0 后权重为 [0,1]", close(w3.reshape(-1), [0.0, 1.0]))
    check("屏蔽 key0 后 context = value1", close(ctx3.reshape(-1), [0.0, 20.0]))


def test_gradients() -> None:
    print("\n== 梯度检验（ε=1e-5，阈值 1e-6）==")
    rng = np.random.default_rng(0)
    B, Tq, S, D = 2, 2, 6, 32
    q = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    k = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    v = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    mask = np.ones((B, S)); mask[0, 4:] = 0.0

    cases = [
        ("加性", BahdanauAttention(D, D, D, attn_dim=16, seed=1)),
        ("点积（零参数）", DotProductAttention(D, D, D, seed=2)),
        ("点积（投影）", DotProductAttention(D, D, D, dim=8, seed=3)),
    ]
    for label, attn in cases:
        qq, kk, vv = q.copy(), k.copy(), v.copy()
        dc = rng.normal(size=(B, Tq, D))
        attn.zero_grad()

        def loss_fn():
            ctx, _ = attn.forward(qq, kk, vv, mask)
            return float((ctx * dc).sum())

        loss_fn()
        attn.zero_grad()
        dinputs = attn.backward(dc)

        tensors = {"query": qq, "keys": kk, "values": vv}
        tensors.update(attn.params())
        analytic = dict(dinputs)
        analytic.update(attn.grads())

        res = check_gradient(tensors, analytic, loss_fn, n_samples=5, eps=1e-5,
                            rng=np.random.default_rng(0))
        summary = summarize_gradient_report(res)
        check(f"{label}: 全部张量梯度通过", summary["passed"],
              f"最大分歧/抖动 {summary['worst_ratio']:.2e}")

    # 反向传播必须**累加**梯度（BPTT 里同一模块会被调用很多次）
    dc = rng.normal(size=(B, Tq, D))

    attn2 = DotProductAttention(D, D, D, dim=8, seed=9)
    attn2.zero_grad()
    attn2.forward(q, k, v, mask); attn2.backward(dc)
    g1 = {kk_: g.copy() for kk_, g in attn2.grads().items()}
    attn2.forward(q, k, v, mask); attn2.backward(dc)
    g2 = attn2.grads()
    check("backward 两次后梯度翻倍（点积投影版）",
          all(close(g2[nm], 2 * g1[nm]) for nm in g1))

    attn3 = BahdanauAttention(D, D, D, attn_dim=16, seed=10)
    attn3.zero_grad()
    attn3.forward(q, k, v, mask); attn3.backward(dc)
    h1 = {kk_: g.copy() for kk_, g in attn3.grads().items()}
    attn3.forward(q, k, v, mask); attn3.backward(dc)
    h2 = attn3.grads()
    check("backward 两次后梯度翻倍（加性版）",
          all(close(h2[nm], 2 * h1[nm]) for nm in h1))

    attn3.zero_grad()
    check("zero_grad 清零", all(bool(np.all(g == 0.0)) for g in attn3.grads().values()))

    # 未 forward 就 backward 必须报错
    fresh = DotProductAttention(D, D, D, dim=8, seed=4)
    try:
        fresh.backward(dc)
        check("未 forward 就 backward 报错", False)
    except RuntimeError:
        check("未 forward 就 backward 报错", True)


def test_gradcheck_catches_bug() -> None:
    print("\n== 负面对照：梯度检验能抓住错误 ==")

    class ForgottenScale(DotProductAttention):
        def backward(self, dcontext):
            saved, self.scale = self.scale, 1.0
            try:
                return super().backward(dcontext)
            finally:
                self.scale = saved

    rng = np.random.default_rng(0)
    B, Tq, S, D = 2, 2, 6, 32
    q = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    k = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    v = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    mask = np.ones((B, S))
    dc = rng.normal(size=(B, Tq, D))

    def run(attn):
        attn.zero_grad()

        def loss_fn():
            ctx, _ = attn.forward(q, k, v, mask)
            return float((ctx * dc).sum())

        loss_fn()
        attn.zero_grad()
        di = attn.backward(dc)
        tensors = {"query": q, "keys": k, "values": v}
        tensors.update(attn.params())
        analytic = dict(di)
        analytic.update(attn.grads())
        res = check_gradient(tensors, analytic, loss_fn, n_samples=4, eps=1e-5,
                            rng=np.random.default_rng(0))
        return summarize_gradient_report(res)["worst_ratio"]

    good = run(DotProductAttention(D, D, D, seed=7))
    bad = run(ForgottenScale(D, D, D, seed=7))
    check("正确实现通过", good < 1e3, f"{good:.2e}")
    check("写错的实现被抓住", bad > 1e6, f"{bad:.2e}")
    check("两者相差 4 个数量级以上", bad / max(good, 1e-30) > 1e4)


def test_real_batch_mask() -> None:
    print("\n== 用真实 batch 的 src_mask（Tq=1 解码形状）==")
    from seq2seq_attn import data as D

    train, _, _ = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(6)))
    rng = np.random.default_rng(0)

    pad_id = train.src_vocab.pad_id
    S = batch.src.shape[1]
    query = np.ascontiguousarray(rng.normal(size=(batch.batch_size, 1, 128)))   # 解码一步：Tq=1
    keys = np.ascontiguousarray(rng.normal(size=(batch.batch_size, S, 128)))

    attn = DotProductAttention(128, 128, 128, seed=0)
    ctx, w = attn(query, keys, keys, batch.src_mask)

    check("输出形状 (B,1,S)", w.shape == (batch.batch_size, 1, S), str(w.shape))
    check("每个样本各自归一化", close(w.sum(-1), 1.0))
    padded_zero = True
    for b in range(batch.batch_size):
        if not np.all(w[b, 0][batch.src_mask[b] < 0.5] == 0.0):
            padded_zero = False
        # <pad> 位置的 key 不应该拿到任何权重
        if np.any(w[b, 0][batch.src[b] == pad_id] != 0.0):
            padded_zero = False
    check("每个样本用的是自己的 mask，padding 权重严格为 0", padded_zero)
    check("无 nan", bool(np.isfinite(w).all() and np.isfinite(ctx).all()))


def main() -> None:
    print("=" * 70)
    print(" 注意力机制测试")
    print("=" * 70)
    test_softmax()
    test_softmax_masked()
    test_shapes_and_masking()
    test_param_counts()
    test_naive_equivalence()
    test_hand_computed()
    test_gradients()
    test_gradcheck_catches_bug()
    test_real_batch_mask()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
