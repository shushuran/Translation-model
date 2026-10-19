"""损失函数、优化器、学习率调度、贪心解码、Seq2Seq 组装 的断言测试。

运行：
    python tests/test_losses_optim.py
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
from seq2seq_attn.losses import CrossEntropyLoss
from seq2seq_attn.model import Seq2Seq, greedy_decode
from seq2seq_attn.nn import check_gradient, log_softmax, softmax, summarize_gradient_report
from seq2seq_attn.optim import Adam, clip_grad_global_norm, collect_params, warmup_cosine_lr

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
def test_log_softmax() -> None:
    print("\n== log_softmax ==")
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    ls = log_softmax(x)
    check("与 log(softmax(x)) 一致（安全区间）", close(ls, np.log(softmax(x)), tol=1e-14))
    check("exp 之和为 1", close(np.exp(ls).sum(-1), 1.0, tol=1e-14))

    # 极端输入不能出现 -inf（这正是 log(softmax) 的毛病）
    extreme = np.array([[1000.0, 0.0, -1000.0]])
    ls_e = log_softmax(extreme)
    check("极端输入无 nan / inf", bool(np.isfinite(ls_e).all()))
    with np.errstate(divide="ignore"):
        naive = np.log(softmax(extreme))
    check("对照：log(softmax(极端)) 会出现 -inf",
          bool(np.isneginf(naive).any()), str(naive))
    check("极端输入下 exp 之和仍为 1", close(np.exp(ls_e).sum(-1), 1.0, tol=1e-14))

    check("平移不变", close(log_softmax(x + 50.0), ls, tol=1e-13))


def test_cross_entropy_forward() -> None:
    print("\n== CrossEntropyLoss 前向 ==")
    rng = np.random.default_rng(0)
    B, T_, V = 3, 4, 6
    logits = rng.normal(size=(B, T_, V))
    targets = rng.integers(0, V, size=(B, T_))
    mask = np.ones((B, T_)); mask[0, 2:] = 0.0

    ce = CrossEntropyLoss()
    loss = ce.forward(logits, targets, mask)
    check("返回标量", isinstance(loss, float))

    # 朴素实现
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    flat = logp.reshape(-1, V); ft = targets.reshape(-1); fm = mask.reshape(-1)
    naive = -(flat[np.arange(len(ft)), ft] * fm).sum() / fm.sum()
    check("与朴素实现一致", abs(naive - loss) < 1e-14, f"{abs(naive - loss):.2e}")

    # 均匀 logits -> loss = log(V)
    uniform = np.zeros((1, 2, V))
    ce2 = CrossEntropyLoss()
    l_unif = ce2.forward(uniform, np.zeros((1, 2), dtype=np.int64))
    check("均匀 logits 时 loss = log(V)", abs(l_unif - np.log(V)) < 1e-14, f"{l_unif:.10f}")

    # 准确率 = argmax 命中目标的比例（用能一眼数清的例子）
    hand = np.zeros((1, 3, V))
    hand[0, 0, 2] = 10.0        # 预测 2，目标 2  -> 对
    hand[0, 1, 5] = 10.0        # 预测 5，目标 5  -> 对
    hand[0, 2, 3] = 10.0        # 预测 3，目标 1  -> 错
    ce_h = CrossEntropyLoss()
    ce_h.forward(hand, np.array([[2, 5, 1]]))
    check("准确率 = 命中数 / 总数", abs(ce_h.last_accuracy - 2.0 / 3.0) < 1e-14,
          f"{ce_h.last_accuracy:.6f}")

    # 完美预测 -> loss ≈ 0
    perfect_logits = np.full((1, 3, V), -50.0)
    for t in range(3):
        perfect_logits[0, t, t] = 50.0
    ce3 = CrossEntropyLoss()
    l_perf = ce3.forward(perfect_logits, np.array([[0, 1, 2]]))
    check("完美预测时 loss ≈ 0", l_perf < 1e-20, f"{l_perf:.2e}")
    check("完美预测时准确率 = 1", ce3.last_accuracy == 1.0)

    # mask 真的排除了 <pad>
    no_mask = CrossEntropyLoss().forward(logits, targets)
    check("加 mask 后 loss 与不加不同", abs(no_mask - loss) > 1e-6,
          f"{no_mask:.6f} vs {loss:.6f}")
    all_one = np.ones((B, T_))
    check("mask 全 1 时与不加 mask 等价",
          abs(CrossEntropyLoss().forward(logits, targets, all_one) - no_mask) < 1e-14)

    # 整批被 mask 掉：不能产生 nan
    ce4 = CrossEntropyLoss()
    l_zero = ce4.forward(logits, targets, np.zeros((B, T_)))
    d_zero = ce4.backward(1.0)
    check("mask 全 0 时 loss = 0", l_zero == 0.0, str(l_zero))
    check("mask 全 0 时梯度全为 0", bool(np.all(d_zero == 0.0)))
    check("mask 全 0 时无 nan", bool(np.isfinite(d_zero).all()))

    # 维度少一点也要能用
    ce5 = CrossEntropyLoss()
    l2d = ce5.forward(rng.normal(size=(5, V)), rng.integers(0, V, size=(5,)))
    check("支持二维 (N,V)", isinstance(l2d, float) and np.isfinite(l2d))

    # 错误输入
    for label, args in [
        ("logits 前置维度与 targets 不符", (logits, targets[:2], None)),
        ("mask 形状不符", (logits, targets, np.ones((B, T_ + 1)))),
        ("target 越界", (logits, np.full((B, T_), V, dtype=np.int64), None)),
        ("logits 一维", (rng.normal(size=(V,)), np.array(0), None)),
    ]:
        try:
            CrossEntropyLoss().forward(*args)
            check(f"拒绝：{label}", False)
        except (ValueError, IndexError):
            check(f"拒绝：{label}", True)

    check("未 forward 就 backward 报错", _raises_runtime(CrossEntropyLoss()))


def _raises_runtime(obj) -> bool:
    try:
        obj.backward(1.0)
        return False
    except RuntimeError:
        return True


def test_cross_entropy_gradient() -> None:
    print("\n== CrossEntropyLoss 反向 ==")
    rng = np.random.default_rng(1)
    B, T_, V = 3, 4, 7
    logits = np.ascontiguousarray(rng.normal(size=(B, T_, V)))
    targets = rng.integers(0, V, size=(B, T_))
    mask = np.ones((B, T_)); mask[0, 3:] = 0.0

    ce = CrossEntropyLoss()
    loss_fn = lambda: ce.forward(logits, targets, mask)

    loss_fn()
    d = ce.backward(1.0)
    check("dlogits 形状与 logits 相同", d.shape == logits.shape)
    res = check_gradient({"logits": logits}, {"logits": d}, loss_fn,
                         n_samples=10, eps=1e-5, rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("梯度检验通过", s["passed"], f"最大分歧/抖动 {s['worst_ratio']:.2e}")

    # 每行梯度之和必须为 0（softmax 输出之和恒为 1，是个约束）
    row_sums = d[1:].sum(axis=-1)          # 第 0 行有 mask，跳过
    check("每行 dlogits 之和 ≈ 0（概率和恒为 1 的约束）",
          bool(np.allclose(row_sums, 0.0, atol=1e-14)), f"max {np.abs(row_sums).max():.2e}")

    # 被 mask 的位置梯度必须为 0
    check("被 mask 的位置梯度为 0", bool(np.all(d[0, 3:, :] == 0.0)))

    # 梯度缩放
    loss_fn()
    d2 = ce.backward(2.5)
    check("dloss 参数线性缩放梯度", bool(np.allclose(d2, 2.5 * d, atol=1e-14)))


def test_adam() -> None:
    print("\n== Adam ==")
    # 最小化 sum(w^2)，梯度 2w
    w = np.array([3.0, -2.0, 0.5])
    g = np.zeros_like(w)
    opt = Adam({"w": w}, {"w": g}, lr=0.1)
    for _ in range(300):
        g[...] = 2.0 * w
        opt.step()
    check("能最小化二次函数", float(np.linalg.norm(w)) < 1e-3, f"|w|={np.linalg.norm(w):.2e}")

    # 与朴素实现对照
    rng = np.random.default_rng(2)
    w1 = rng.normal(size=(4, 5))
    w2 = w1.copy()
    g1 = np.zeros_like(w1)
    opt1 = Adam({"w": w1}, {"w": g1}, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    m = np.zeros_like(w1); v = np.zeros_like(w1)
    for t in range(1, 21):
        grad = rng.normal(size=(4, 5))
        g1[...] = grad
        opt1.step()
        # 参考实现
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        w2 -= 0.01 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    check("与朴素 Adam 实现一致", close(w1, w2, tol=1e-14), f"最大差 {np.abs(w1 - w2).max():.2e}")

    # 偏差修正：把首步更新量从 3.16·lr 压回恰好 1·lr
    #   无修正时 m/√v = (1-β₁)g / (√(1-β₂)·|g|) = 0.1/√0.001 = 3.162
    wb = np.zeros((1,))
    gb = np.zeros((1,))
    optb = Adam({"w": wb}, {"w": gb}, lr=1.0, bias_correction=True)
    gb[...] = 5.0
    optb.step()
    check("有偏差修正时首步更新量 ≈ lr", abs(abs(wb[0]) - 1.0) < 1e-6, f"{wb[0]:.6f}")

    wc = np.zeros((1,))
    gc = np.zeros((1,))
    optc = Adam({"w": wc}, {"w": gc}, lr=1.0, bias_correction=False)
    gc[...] = 5.0
    optc.step()
    expected_ratio = (1 - 0.9) / np.sqrt(1 - 0.999)
    check("无偏差修正时首步更新量偏**大**（这正是要修正的问题）",
          abs(abs(wc[0]) - expected_ratio) < 1e-6,
          f"{wc[0]:.6f}（理论值 {expected_ratio:.6f}）")
    check("两种设定的首步差异约 3.16 倍", abs(abs(wc[0] / wb[0]) - expected_ratio) < 1e-6)

    # zero_grad / state_dict
    wd = np.zeros((2,)); gd = np.ones((2,))
    optd = Adam({"w": wd}, {"w": gd}, lr=0.1)
    optd.step()
    sd = optd.state_dict()
    check("state_dict 含 t/lr/m/v", set(sd) == {"t", "lr", "m", "v"}, str(set(sd)))
    optd.zero_grad()
    check("zero_grad 清零", bool(np.all(gd == 0.0)))
    wd2 = np.zeros((2,)); gd2 = np.zeros((2,))
    opte = Adam({"w": wd2}, {"w": gd2}, lr=0.1)
    opte.load_state_dict(sd)
    check("load_state_dict 恢复 t", opte.t == sd["t"])
    check("load_state_dict 恢复矩估计",
          close(opte.m["w"], sd["m"]["w"]) and close(opte.v["w"], sd["v"]["w"]))

    # 形状不匹配的键会被跳过（而不是静默出错）
    wf = np.zeros((3,)); gf = np.zeros((5,))
    optf = Adam({"w": wf}, {"w": gf}, lr=0.1)
    check("形状不匹配的键被跳过", optf.skipped == ["w"] and len(optf._keys) == 0, str(optf.skipped))

    try:
        Adam({"a": np.zeros(2)}, {"b": np.zeros(2)})
        check("缺失梯度时报错", False)
    except KeyError:
        check("缺失梯度时报错", True)

    check("lr 可动态设置", _lr_settable())


def _lr_settable() -> bool:
    w = np.zeros(1); g = np.ones(1)
    opt = Adam({"w": w}, {"w": g}, lr=0.1)
    opt.set_lr(0.5)
    opt.step()
    return abs(abs(w[0]) - 0.5) < 1e-6


def test_clip_grad() -> None:
    print("\n== 全局范数梯度裁剪 ==")
    rng = np.random.default_rng(3)
    params = {"a": np.zeros((3, 4)), "b": np.zeros((2,))}
    grads = {"a": rng.normal(size=(3, 4)), "b": rng.normal(size=(2,))}

    expected = float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))
    before = {k: v.copy() for k, v in grads.items()}
    norm = clip_grad_global_norm(params, grads, max_norm=1.0)
    check("返回**裁剪前**的范数", abs(norm - expected) < 1e-12, f"{norm:.6f} vs {expected:.6f}")

    after = float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))
    check("裁剪后总范数 = max_norm", abs(after - 1.0) < 1e-6, f"{after:.8f}")

    # 方向必须保持（统一缩放）
    ratio = grads["a"] / np.where(before["a"] == 0, 1e-300, before["a"])
    check("各元素按同一比例缩放（方向不变）",
          float(np.std(ratio)) < 1e-10, f"std={np.std(ratio):.2e}")

    # 不超阈值时不动
    grads2 = {"a": rng.normal(size=(3, 4)) * 0.01, "b": rng.normal(size=(2,)) * 0.01}
    snap = {k: v.copy() for k, v in grads2.items()}
    clip_grad_global_norm(params, grads2, max_norm=10.0)
    check("范数未超阈值时不改动", close(grads2["a"], snap["a"]) and close(grads2["b"], snap["b"]))

    # max_norm=0 表示不裁剪
    grads3 = {"a": rng.normal(size=(3, 4)) * 100}
    snap3 = grads3["a"].copy()
    clip_grad_global_norm(params, grads3, max_norm=0.0)
    check("max_norm = 0 表示不裁剪", close(grads3["a"], snap3))


def test_lr_schedule() -> None:
    print("\n== warmup + 余弦退火 ==")
    base, total = 2e-3, 3000
    lrs = [warmup_cosine_lr(s, total, base) for s in range(total)]

    check("首步很小（不直接上大学习率）", lrs[0] < base * 0.01, f"{lrs[0]:.2e}")
    warmup_steps = int(total * 0.1)
    check("warmup 段单调递增", all(lrs[i] <= lrs[i + 1] + 1e-15 for i in range(warmup_steps - 1)))
    check("warmup 结束时达到峰值", abs(lrs[warmup_steps - 1] - base) < 1e-12,
          f"{lrs[warmup_steps - 1]:.3e}")
    check("退火段单调递减", all(lrs[i] >= lrs[i + 1] - 1e-15 for i in range(warmup_steps, total - 1)))
    check("末步降到 min_lr_ratio × base", abs(lrs[-1] - base * 0.1) < 1e-9, f"{lrs[-1]:.3e}")
    check("全局峰值就是 base_lr", abs(max(lrs) - base) < 1e-12)

    check("total_steps=1 不崩", warmup_cosine_lr(0, 1, 1e-3) > 0)
    check("warmup_frac=0 直接进余弦段", warmup_cosine_lr(0, 100, 1e-3, warmup_frac=0.0) == 1e-3)
    check("step 超出 total 时被夹住",
          abs(warmup_cosine_lr(99999, total, base) - base * 0.1) < 1e-9)


def test_collect_params() -> None:
    print("\n== 参数收集 ==")
    train, _, _ = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CONFIG.model)
    params, grads = collect_params(model)
    check("参数与梯度键一致", set(params) == set(grads), str(set(params) ^ set(grads)))
    check("参数数量 = 模型参数量之和",
          sum(p.size for p in params.values()) == model.num_params(),
          f"{sum(p.size for p in params.values())} vs {model.num_params()}")
    check("键带 enc./dec. 前缀",
          any(k.startswith("enc.") for k in params) and any(k.startswith("dec.") for k in params))
    check("包含编码器 embedding", "enc.emb.weight" in params)
    check("包含注意力打分参数", "dec.attn.W_q" in params)
    check("包含输出层", "dec.out.W" in params)


def test_seq2seq_and_greedy() -> None:
    print("\n== Seq2Seq 组装 与 贪心解码 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CONFIG.model)
    batch = D.collate(train, list(range(4)))

    check("参数量 = 编码器 + 解码器",
          model.num_params() == model.enc.num_params() + model.dec.num_params())

    loss, logits, weights = model.forward(batch)
    check("loss 是有限标量", np.isfinite(loss))
    check("logits 形状", logits.shape == (4, batch.tgt_in.shape[1], len(train.tgt_vocab)))
    check("weights 形状", weights.shape == (4, batch.tgt_in.shape[1], batch.src.shape[1]))

    model.zero_grad()
    model.backward()
    pg = model.named_grads()
    nonzero = {k: float(np.linalg.norm(v)) for k, v in pg.items() if v.size}
    check("每个参数都收到非零梯度", all(v > 0 for v in nonzero.values()),
          str({k: f"{v:.2e}" for k, v in list(nonzero.items())[:4]}))
    check("编码器也收到梯度", any(k.startswith("enc.") and np.linalg.norm(v) > 0
                                 for k, v in pg.items()))

    # state_dict 往返
    sd = model.state_dict()
    check("state_dict 是副本（改它不影响模型）",
          not np.shares_memory(sd["dec.out.W"], model.dec.out.params()["W"]))
    ref_loss, _, _ = model.forward(batch)
    backup = {k: v.copy() for k, v in sd.items()}
    for p in model.named_parameters_dict().values():
        p[...] = 0.0
    zero_loss, zero_logits, _ = model.forward(batch)
    # 全零参数 -> logits 恒为 0 -> 均匀分布 -> loss 恰好是 log(V)
    check("全零参数时 logits 恒为 0", bool(np.all(zero_logits == 0.0)))
    check("全零参数时 loss = log(V)",
          abs(zero_loss - np.log(len(train.tgt_vocab))) < 1e-12,
          f"{zero_loss:.12f} vs {np.log(len(train.tgt_vocab)):.12f}")
    check("清零确实改变了 loss", abs(zero_loss - ref_loss) > 1e-9,
          f"|Δ|={abs(zero_loss - ref_loss):.3e}")
    model.load_state_dict(backup)
    restored_loss, _, _ = model.forward(batch)
    check("load_state_dict 恢复后 loss 逐位一致",
          restored_loss == ref_loss, f"{restored_loss!r} vs {ref_loss!r}")

    try:
        model.load_state_dict({"x": np.zeros(1)})
        check("键不匹配时报错", False)
    except KeyError:
        check("键不匹配时报错", True)

    # ---- 贪心解码 ----
    max_len = test.max_tgt_len + 2
    seqs, ws, lens = greedy_decode(model, batch.src, batch.src_mask, max_len,
                                   bos_id=train.tgt_vocab.bos_id,
                                   eos_id=train.tgt_vocab.eos_id,
                                   pad_id=train.tgt_vocab.pad_id)
    check("序列形状 (B, T')", seqs.shape[0] == 4 and seqs.shape[1] <= max_len, str(seqs.shape))
    check("权重形状与序列对齐", ws.shape[:2] == seqs.shape, str(ws.shape))
    check("长度在 [1, T'] 之间", bool(np.all(lens >= 1) and np.all(lens <= seqs.shape[1])),
          str(lens))
    check("输出无 nan", bool(np.isfinite(ws).all()))

    # 每个样本：<eos> 之后必须全是 <pad>
    pad_id = train.tgt_vocab.pad_id
    eos_id = train.tgt_vocab.eos_id
    ok = True
    for b in range(4):
        row = seqs[b]
        pos = np.where(row == eos_id)[0]
        if pos.size:
            after = row[pos[0] + 1:]
            if np.any(after != pad_id):
                ok = False
    check("<eos> 之后全是 <pad>", ok)

    # 长度必须与「第一个 <eos> 的位置」一致
    ok = True
    for b in range(4):
        pos = np.where(seqs[b] == eos_id)[0]
        expect = int(pos[0]) + 1 if pos.size else seqs.shape[1]
        if int(lens[b]) != expect:
            ok = False
    check("长度 = 第一个 <eos> 的位置 + 1", ok)

    # 每步权重行和为 1（只在有效步上检查）
    row_sums = ws.sum(-1)
    check("注意力权重行和为 1",
          bool(np.allclose(row_sums, 1.0, atol=1e-12)), f"max偏差 {np.abs(row_sums - 1).max():.2e}")


def test_greedy_deterministic_and_causal() -> None:
    print("\n== 贪心解码的确定性与因果性 ==")
    train, _, test = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CONFIG.model)
    batch = D.collate(test, list(range(3)))
    kwargs = dict(bos_id=train.tgt_vocab.bos_id, eos_id=train.tgt_vocab.eos_id,
                  pad_id=train.tgt_vocab.pad_id)
    a, _, _ = greedy_decode(model, batch.src, batch.src_mask, 12, **kwargs)
    b, _, _ = greedy_decode(model, batch.src, batch.src_mask, 12, **kwargs)
    check("两次结果完全相同", bool(np.array_equal(a, b)))

    # 独立性：只解码第 2 条，结果应与整批解码的第 2 行一致
    one, _, _ = greedy_decode(model, batch.src[1:2], batch.src_mask[1:2], 12, **kwargs)
    check("逐样本解码与批量解码结果一致",
          bool(np.array_equal(one[0], a[1])), f"{one[0]} vs {a[1]}")

    # 一开始就强制生成 <eos> 的模型：输出应是 [<eos>, <pad>, ...]
    orig = model.dec.decode_step

    def force_eos(prev_token, src_h, src_mask, s_prev):
        logits, w, s = orig(prev_token, src_h, src_mask, s_prev)
        logits = np.full_like(logits, -50.0)
        logits[:, train.tgt_vocab.eos_id] = 50.0
        return logits, w, s

    model.dec.decode_step = force_eos
    seqs, _, lens = greedy_decode(model, batch.src, batch.src_mask, 12, **kwargs)
    model.dec.decode_step = orig
    check("模型只会输出 <eos> 时立即停止",
          bool(np.all(seqs[:, 0] == train.tgt_vocab.eos_id)) and bool(np.all(lens == 1)), str(lens))
    check("后续位置全部补 <pad>",
          bool(np.all(seqs[:, 1:] == train.tgt_vocab.pad_id)))

    try:
        greedy_decode(model, batch.src, batch.src_mask, 0, **kwargs)
        check("max_len=0 报错", False)
    except ValueError:
        check("max_len=0 报错", True)


def test_training_smoke() -> None:
    print("\n== 训练冒烟测试：loss 必须下降 ==")
    train, dev, _ = D.build_datasets(CONFIG.data)
    small = train.subset(16)
    model = T.build_model(train, CONFIG.model)
    params, grads, opt = T._make_optimizer(model, CONFIG.train)
    batch = D.collate(small, list(range(16)))

    losses = []
    for step in range(120):
        opt.zero_grad()
        loss, _, _ = model.forward(batch)
        model.backward()
        clip_grad_global_norm(params, grads, CONFIG.train.grad_clip)
        opt.step()
        losses.append(loss)

    check("loss 明显下降", losses[-1] < losses[0] * 0.3,
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("loss 单调趋势（后半段均值 < 前半段）",
          np.mean(losses[-30:]) < np.mean(losses[:30]))
    check("loss 无 nan", all(np.isfinite(l) for l in losses))
    ev = T.evaluate_teacher_forcing(model, small, batch_size=16)
    check("16 条样本的 token 准确率提升", ev["token_acc"] > 0.8, f"{ev['token_acc']:.4f}")

    # 健全性检查函数本身
    r = T.sanity_check_overfit(CONFIG, n_examples=6, steps=60, verbose=False)
    check("sanity_check_overfit 返回完整结构",
          {"model", "loss_history", "final", "greedy"} <= set(r), str(set(r)))
    check("sanity check 的 loss 也在下降", r["final_loss"] < r["first_loss"])


def test_evaluate_functions() -> None:
    print("\n== 评估函数 ==")
    train, dev, _ = D.build_datasets(CONFIG.data)
    model = T.build_model(train, CONFIG.model)
    small = dev.subset(32)
    tf = T.evaluate_teacher_forcing(model, small, batch_size=16)
    check("TF 返回四个字段",
          set(tf) == {"loss", "token_acc", "seq_acc", "n_seq"}, str(set(tf)))
    check("TF 的 n_seq = 样本数", tf["n_seq"] == 32, str(tf["n_seq"]))
    check("未训练时准确率不会超过 1", 0.0 <= tf["token_acc"] <= 1.0)

    ar = T.evaluate_greedy(model, small, max_len=small.max_tgt_len + 2, batch_size=16)
    check("AR 返回三个字段", set(ar) == {"token_acc", "seq_acc", "n_seq"}, str(set(ar)))
    check("AR 的 n_seq = 样本数", ar["n_seq"] == 32)
    check("随机初始化的模型：准确率很低", ar["seq_acc"] < 0.2, f"{ar['seq_acc']:.4f}")

    rows = T.decode_sentences(model, small, range(3), small.max_tgt_len + 2)
    check("decode_sentences 返回 3 条", len(rows) == 3)
    check("每条含 src/ref/tf/greedy",
          {"src", "ref", "tf", "greedy", "weight", "src_tokens"} <= set(rows[0]),
          str(set(rows[0])))
    check("ref 与数据集的 tgt_text 一致",
          rows[0]["ref"] == small.examples[0].tgt_text)
    check("weight 形状 = 源句长度",
          rows[0]["weight"].shape[1] == len(small.examples[0].src_tokens))


def main() -> None:
    print("=" * 70)
    print(" 损失 / 优化器 / 训练 测试")
    print("=" * 70)
    test_log_softmax()
    test_cross_entropy_forward()
    test_cross_entropy_gradient()
    test_adam()
    test_clip_grad()
    test_lr_schedule()
    test_collect_params()
    test_seq2seq_and_greedy()
    test_greedy_deterministic_and_causal()
    test_training_smoke()
    test_evaluate_functions()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
