"""解码器（带注意力）与梯度检验判据的断言测试。

运行：
    python tests/test_decoder.py
"""

from __future__ import annotations

import dataclasses
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
from seq2seq_attn.attention import BahdanauAttention, DotProductAttention
from seq2seq_attn.config import CONFIG
from seq2seq_attn.decoder import INIT_MODES, OUTPUT_MODES, AttentionDecoder
from seq2seq_attn.encoder import GRUEncoder
from seq2seq_attn.nn import Linear, check_gradient, summarize_gradient_report

PASSED = 0
V_SRC, V_TGT = 13, 19
CFG = CONFIG.model


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
def make(seed=0, **kw):
    enc = GRUEncoder(V_SRC, CFG.src_emb_dim, CFG.hidden_dim,
                     init_scale=CFG.init_scale, seed=seed)
    dec = AttentionDecoder(V_TGT, CFG.tgt_emb_dim, CFG.hidden_dim, CFG.hidden_dim, CFG,
                           output_mode=kw.get("output_mode", CFG.decoder_output),
                           init_mode=kw.get("init_mode", CFG.decoder_init),
                           init_scale=CFG.init_scale, seed=seed)
    return enc, dec


def sample(B=2, S=6, T=5, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, V_SRC, size=(B, S))
    mask = np.ones((B, S))
    mask[0, 4:] = 0.0
    tgt_in = rng.integers(0, V_TGT, size=(B, T))
    return src, mask, tgt_in


def decoder_loss(enc, dec, src, mask, tgt_in, G):
    def loss_fn():
        hs, hf = enc.forward(src, mask)
        lg, _ = dec.forward(tgt_in, hs, mask, hf)
        return float((lg * G).sum())
    return loss_fn


# --------------------------------------------------------------------------------------
def test_shapes() -> None:
    print("\n== 形状与基本行为 ==")
    enc, dec = make()
    src, mask, tgt_in = sample()
    hs, hf = enc.forward(src, mask)
    logits, weights = dec.forward(tgt_in, hs, mask, hf)

    B, S, T = tgt_in.shape[0], hs.shape[1], tgt_in.shape[1]
    check("logits 形状 (B,T,V)", logits.shape == (B, T, V_TGT), str(logits.shape))
    check("weights 形状 (B,T,S)", weights.shape == (B, T, S), str(weights.shape))
    check("输出无 nan", bool(np.isfinite(logits).all() and np.isfinite(weights).all()))
    check("注意力权重每行和为 1", close(weights.sum(-1), 1.0, tol=1e-12))
    check("padding 位置权重恒为 0",
          bool(np.all(weights[np.broadcast_to(mask[:, None, :] < 0.5, weights.shape)] == 0.0)))
    check("非 padding 位置权重为正",
          bool(np.all(weights[np.broadcast_to(mask[:, None, :] > 0.5, weights.shape)] > 0.0)))

    s0, _ = dec.initial_state(hf)
    check("s₀ 形状 (B,H)", s0.shape == (B, CFG.hidden_dim), str(s0.shape))

    # 桥接是 tanh 投影，不是恒等
    check("bridge 模式下 s₀ ≠ h_final", not close(s0, hf, tol=1e-6))

    # 参数拆分相加一致
    br = dec.num_params_breakdown()
    check("参数量拆分相加一致",
          sum(v for k, v in br.items() if k != "total") == br["total"], str(br))
    check("子模块齐全", set(br) == {"emb", "attn", "cell", "bridge", "out", "total"}, str(set(br)))


def test_teacher_forcing_and_causality() -> None:
    print("\n== teacher forcing 错位与因果性 ==")
    train, _, _ = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(4)))
    vocab = train.tgt_vocab

    # <bos> + y[:-1] 与 y + <eos> 错位
    width = batch.tgt_in.shape[1]
    pos = np.arange(width - 1)[None, :]
    valid = pos < (batch.tgt_len - 1)[:, None]
    check("tgt_in / tgt_out 错位（数据管道保证）",
          bool(np.array_equal(batch.tgt_in[:, 1:][valid], batch.tgt_out[:, :-1][valid])))

    enc, dec = make()
    src, mask, tgt_in = sample(B=3, S=7, T=6, seed=1)
    hs, hf = enc.forward(src, mask)
    logits1, _ = dec.forward(tgt_in, hs, mask, hf)

    ok = True
    worst = 0.0
    rng = np.random.default_rng(9)
    for cut in range(tgt_in.shape[1]):
        mutated = tgt_in.copy()
        if cut + 1 < tgt_in.shape[1]:
            mutated[:, cut + 1:] = rng.integers(0, V_TGT, size=(3, tgt_in.shape[1] - cut - 1))
        logits2, _ = dec.forward(mutated, hs, mask, hf)
        d = float(np.abs(logits1[:, : cut + 1, :] - logits2[:, : cut + 1, :]).max())
        worst = max(worst, d)
        if d != 0.0:
            ok = False
    check("篡改 t 之后的输入不影响前 t+1 步输出（因果性）", ok, f"最大差 {worst:.3e}")

    # 反过来：篡改第 t 步的输入**必须**影响第 t 步输出（否则模型没用上输入）
    mutated = tgt_in.copy()
    mutated[:, 2] = (mutated[:, 2] + 1) % V_TGT
    logits3, _ = dec.forward(mutated, hs, mask, hf)
    check("篡改第 2 步输入会改变第 2 步输出",
          float(np.abs(logits1[:, 2, :] - logits3[:, 2, :]).max()) > 1e-9)


def test_output_and_init_modes() -> None:
    print("\n== output_mode / init_mode 兼容性 ==")
    src, mask, tgt_in = sample(B=2, S=5, T=4, seed=2)
    for mode in OUTPUT_MODES:
        enc, dec = make(output_mode=mode)
        hs, hf = enc.forward(src, mask)
        logits, _ = dec.forward(tgt_in, hs, mask, hf)
        expect = {"s": CFG.hidden_dim,
                  "s_c": CFG.hidden_dim + CFG.hidden_dim,
                  "s_c_emb": CFG.hidden_dim + CFG.hidden_dim + CFG.tgt_emb_dim}[mode]
        check(f"output_mode={mode}: 输出层输入维度", dec.output_input_dim() == expect, str(expect))
        check(f"output_mode={mode}: logits 形状", logits.shape == (2, 4, V_TGT))
        # 切片定义必须与拼装一致
        sl = dec.output_logit_slices()
        total = dec.output_input_dim()
        spans = [sl[k] for k in ("s", "ctx", "emb") if sl[k] is not None]
        check(f"output_mode={mode}: 切片连续覆盖", spans[0][0] == 0 and spans[-1][1] == total,
              str(spans))

    for mode in INIT_MODES:
        enc, dec = make(init_mode=mode)
        hs, hf = enc.forward(src, mask)
        s0, bc = dec.initial_state(hf)
        if mode == "zero":
            check("init_mode=zero: s₀ 为零", close(s0, np.zeros_like(s0)))
        elif mode == "direct":
            check("init_mode=direct: s₀ == h_final", close(s0, hf))
        else:
            check("init_mode=bridge: s₀ 由 tanh 投影得到", bc is not None)
        logits, _ = dec.forward(tgt_in, hs, mask, hf)
        check(f"init_mode={mode}: 能正常前向", logits.shape == (2, 4, V_TGT))

    try:
        AttentionDecoder(V_TGT, 16, 8, 6, CFG, output_mode="nope", seed=0)
        check("拒绝非法 output_mode", False)
    except ValueError:
        check("拒绝非法 output_mode", True)
    try:
        AttentionDecoder(V_TGT, 16, 8, 6, CFG, init_mode="nope", seed=0)
        check("拒绝非法 init_mode", False)
    except ValueError:
        check("拒绝非法 init_mode", True)
    try:
        AttentionDecoder(V_TGT, 16, 8, 6, CFG, init_mode="direct", seed=0).initial_state(
            np.zeros((1, 6)))
        check("reject：direct 但维度不匹配", False)
    except ValueError:
        check("reject：direct 但维度不匹配", True)

    # 注意力类型可切换
    for attn_type, cls in (("bahdanau", BahdanauAttention), ("dot", DotProductAttention)):
        cfg2 = dataclasses.replace(CFG, attn_type=attn_type)
        dec = AttentionDecoder(V_TGT, CFG.tgt_emb_dim, CFG.hidden_dim, CFG.hidden_dim, cfg2, seed=0)
        check(f"attn_type={attn_type} 生效", isinstance(dec.attn, cls), type(dec.attn).__name__)


def test_errors() -> None:
    print("\n== 错误输入应当大声报错 ==")
    enc, dec = make()
    src, mask, tgt_in = sample()
    hs, hf = enc.forward(src, mask)
    dec.forward(tgt_in, hs, mask, hf)

    try:
        dec.backward(np.zeros((2, 5, V_TGT + 1)))
        check("dlogits 形状错误报错", False)
    except ValueError:
        check("dlogits 形状错误报错", True)
    try:
        AttentionDecoder(V_TGT, 8, 6, 6, CFG, seed=0).backward(np.zeros((2, 5, V_TGT)))
        check("未 forward 就 backward 报错", False)
    except RuntimeError:
        check("未 forward 就 backward 报错", True)
    try:
        dec.forward(tgt_in[0], hs, mask, hf)
        check("tgt_in 非二维报错", False)
    except ValueError:
        check("tgt_in 非二维报错", True)
    try:
        dec.forward(tgt_in, hs, mask[:, :-1], hf)
        check("mask 形状不匹配报错", False)
    except ValueError:
        check("mask 形状不匹配报错", True)


def test_decoder_gradients() -> None:
    print("\n== 解码器梯度检验 ==")
    enc, dec = make(seed=3)
    src, mask, tgt_in = sample(B=2, S=5, T=4, seed=3)
    hs, hf = enc.forward(src, mask)
    G = np.random.default_rng(3).normal(size=(2, 4, V_TGT))
    loss_fn = decoder_loss(enc, dec, src, mask, tgt_in, G)

    loss_fn()
    dec.zero_grad()
    d = dec.backward(G)
    check("返回 src_h 梯度", d["src_h"].shape == hs.shape, str(d["src_h"].shape))
    check("返回 src_h_final 梯度", d["src_h_final"].shape == hf.shape, str(d["src_h_final"].shape))

    res = check_gradient(dec.named_parameters_dict(), dec.named_grads(), loss_fn,
                         n_samples=3, eps=1e-5, rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("解码器全部参数通过", s["passed"], f"最大分歧/抖动 {s['worst_ratio']:.2e}")
    statuses = {r["status"] for r in res}
    check("状态标记合理（ok 或 low-snr）", statuses <= {"ok", "low-snr"}, str(statuses))

    # 多次 backward 累加（backward 自己不清零，调用方负责）
    enc2, dec2 = make(seed=3)
    src2, mask2, tgt_in2 = sample(B=2, S=5, T=4, seed=3)
    hs2, hf2 = enc2.forward(src2, mask2)
    dec2.zero_grad()
    dec2.forward(tgt_in2, hs2, mask2, hf2); dec2.backward(G)
    g1 = {k: v.copy() for k, v in dec2.named_grads().items()}
    dec2.forward(tgt_in2, hs2, mask2, hf2); dec2.backward(G)
    check("backward 两次梯度翻倍",
          all(close(dec2.named_grads()[k], 2 * g1[k]) for k in g1))
    dec2.zero_grad()
    check("zero_grad 递归清零",
          all(bool(np.all(v == 0)) for v in dec2.named_grads().values()))


def test_end_to_end_chain() -> None:
    print("\n== 端到端链路：梯度必须传到编码器 ==")
    enc, dec = make(seed=4)
    src, mask, tgt_in = sample(B=2, S=5, T=4, seed=4)
    G = np.random.default_rng(4).normal(size=(2, 4, V_TGT))
    loss_fn = decoder_loss(enc, dec, src, mask, tgt_in, G)

    # 编码器每个参数都必须收到非零梯度
    loss_fn()
    enc.zero_grad(); dec.zero_grad()
    hs, hf = enc.forward(src, mask)
    lg, _ = dec.forward(tgt_in, hs, mask, hf)
    d = dec.backward(G)
    enc.backward(d["src_h"], d["src_h_final"])
    enc_g = enc.named_grads()
    nonzero = {k: float(np.linalg.norm(v)) for k, v in enc_g.items() if v.size}
    check("编码器全部参数梯度非零", all(v > 0 for v in nonzero.values()),
          str({k: f"{v:.3e}" for k, v in nonzero.items()}))

    # 整条链路联合梯度检验
    def analytic():
        enc.zero_grad(); dec.zero_grad()
        hs2, hf2 = enc.forward(src, mask)
        lg2, _ = dec.forward(tgt_in, hs2, mask, hf2)
        dd = dec.backward(G)
        enc.backward(dd["src_h"], dd["src_h_final"])
        out = {f"enc.{k}": v for k, v in enc.named_grads().items()}
        out.update({f"dec.{k}": v for k, v in dec.named_grads().items()})
        return out

    loss_fn()
    ana = analytic()
    tensors = {f"enc.{k}": v for k, v in enc.named_parameters_dict().items()}
    tensors.update({f"dec.{k}": v for k, v in dec.named_parameters_dict().items()})
    res = check_gradient(tensors, ana, loss_fn, n_samples=3, eps=1e-5,
                         rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("编码器 + 解码器 联合梯度检验通过", s["passed"],
          f"最大分歧/抖动 {s['worst_ratio']:.2e}")
    check("编码器参数也参与了检验",
          sum(1 for r in res if r["name"].startswith("enc.")) == len(list(enc.named_parameters())))
    check("检验覆盖 ≥ 15 个张量", s["n_tensors"] >= 15, str(s["n_tensors"]))


def test_missing_value_grad_regression() -> None:
    print("\n== 回归测试：漏掉 dkeys + dvalues 相加 ==")
    print("  （values = keys = H_seq，两者是同一份张量，梯度必须叠加。）")

    class MissingValueGrad(AttentionDecoder):
        def backward(self, dlogits):
            c = self._cache
            B, T, S = c["B"], c["T"], c["S"]
            Hs, E = self.src_hidden_dim, self.emb_dim
            self.zero_grad()
            dsrc_h = np.zeros((B, S, Hs))
            demb = np.zeros((B, T, E))
            carry = np.zeros((B, self.hidden_dim))
            for t in reversed(range(T)):
                self.out.load_cache(c["out_caches"][t])
                ds_t, dctx, demb_o = self._split_output_grad(self.out.backward(dlogits[:, t, :]))
                ds_t = ds_t + carry
                if demb_o is not None:
                    demb[:, t, :] += demb_o
                self.cell.load_cache(c["cell_caches"][t])
                dc = self.cell.backward(ds_t)
                demb[:, t, :] += dc["x"][:, :E]
                dctx = dctx + dc["x"][:, E:]
                prev = dc["h"]
                self.attn.load_cache(c["attn_caches"][t])
                da = self.attn.backward(dctx[:, None, :])
                prev = prev + da["query"][:, 0, :]
                dsrc_h += da["keys"]                 # ← 少了 + da["values"]
                carry = prev
            self.emb.backward(demb)
            bc = c["bridge_cache"]
            dhf = np.zeros((B, Hs))
            if bc is not None:
                s0 = bc["s0"]
                dhf = self.bridge.backward(carry * (1 - s0 * s0))
            elif self.init_mode == "direct":
                dhf = carry
            return {"src_h": dsrc_h, "src_h_final": dhf}

    def run(dec_cls):
        enc = GRUEncoder(V_SRC, CFG.src_emb_dim, CFG.hidden_dim, seed=0)
        dec = dec_cls(V_TGT, CFG.tgt_emb_dim, CFG.hidden_dim, CFG.hidden_dim, CFG, seed=0)
        src, mask, tgt_in = sample(B=2, S=5, T=4, seed=5)
        G = np.random.default_rng(5).normal(size=(2, 4, V_TGT))
        loss_fn = decoder_loss(enc, dec, src, mask, tgt_in, G)

        def ana():
            enc.zero_grad(); dec.zero_grad()
            hs, hf = enc.forward(src, mask)
            lg, _ = dec.forward(tgt_in, hs, mask, hf)
            dd = dec.backward(G)
            enc.backward(dd["src_h"], dd["src_h_final"])
            out = {f"enc.{k}": v for k, v in enc.named_grads().items()}
            out.update({f"dec.{k}": v for k, v in dec.named_grads().items()})
            return out

        loss_fn()
        a = ana()
        tensors = {f"enc.{k}": v for k, v in enc.named_parameters_dict().items()}
        tensors.update({f"dec.{k}": v for k, v in dec.named_parameters_dict().items()})
        return summarize_gradient_report(
            check_gradient(tensors, a, loss_fn, n_samples=3, eps=1e-5,
                           rng=np.random.default_rng(0)))

    good = run(AttentionDecoder)
    bad = run(MissingValueGrad)

    check("正确实现通过", good["passed"], f"最大分歧/抖动 {good['worst_ratio']:.2e}")
    check("漏掉一半的实现被判 FAIL", not bad["passed"], f"不通过项：{bad['failed'][:3]}")
    bad_enc = [n for n in bad["failed"] if n.startswith("enc.")]
    bad_dec = [n for n in bad["failed"] if n.startswith("dec.")]
    check("失败项集中在编码器", len(bad_enc) > 0 and len(bad_dec) == 0,
          f"编码器 {len(bad_enc)} 项，解码器 {len(bad_dec)} 项")
    print(f"        （这正是这个 bug 危险的地方：只检验解码器参数的实现会全部通过。）")


def test_gradcheck_criterion() -> None:
    print("\n== 梯度检验判据本身的性质 ==")
    print("  判据：|解析-数值| ≤ 10·( |数值(ε)-数值(4ε)| + 1e-6·|解析| )")
    print("  返回的「分歧/抖动」= |解析-数值| / max(抖动, 噪声地板)。")

    rng = np.random.default_rng(0)
    lin = Linear(6, 4, seed=1)
    x = np.ascontiguousarray(rng.normal(size=(3, 6)))
    G = rng.normal(size=(3, 4))

    def loss_fn():
        return float((lin.forward(x) * G).sum())

    loss_fn()
    lin.zero_grad()
    dx = lin.backward(G)
    tensors = {"x": x}; tensors.update(lin.params())
    analytic = {"x": dx}; analytic.update(lin.grads())

    # 1) 正确实现在多个 ε 下都通过，且比值是 O(1)
    for eps in (1e-6, 1e-5, 1e-4):
        res = check_gradient(tensors, analytic, loss_fn, n_samples=4, eps=eps,
                             rng=np.random.default_rng(0))
        s = summarize_gradient_report(res)
        check(f"ε={eps:.0e}: 正确实现通过", s["passed"])
        check(f"ε={eps:.0e}: 分歧/抖动 < 1e3", s["worst_ratio"] < 1e3,
              f"{s['worst_ratio']:.2e}")

    # 2) 解析梯度偏 0.1% 也要被抓住
    for factor, label in ((1.001, "偏大 0.1%"), (0.999, "偏小 0.1%"), (0.0, "全零")):
        wrong = {k: (v * factor if k != "x" else v) for k, v in analytic.items()}
        res = check_gradient(tensors, wrong, loss_fn, n_samples=4, eps=1e-5,
                             rng=np.random.default_rng(0))
        s = summarize_gradient_report(res)
        check(f"解析梯度{label} -> 被判 FAIL", not s["passed"],
              f"不通过项 {s['failed']}")

    # 3) 形状不符必须报错（不能静默按位置比较）
    try:
        check_gradient({"x": x}, {"x": np.zeros((6, 3))}, loss_fn, n_samples=2)
        check("解析梯度形状不符报错", False)
    except ValueError:
        check("解析梯度形状不符报错", True)

    # 4) 非连续张量必须报错（否则就地扰动会作用在副本上，静默测不出东西）
    noncontig = np.asfortranarray(rng.normal(size=(3, 6)))
    try:
        check_gradient({"x": noncontig}, {"x": np.zeros((3, 6))}, loss_fn, n_samples=2)
        check("非 C 连续张量报错", False)
    except ValueError:
        check("非 C 连续张量报错", True)


def test_decoder_low_snr_regression() -> None:
    print("\n== 回归测试：解码器 attn.W_q 的低信噪比情形 ==")
    print("  （初始化解码器隐状态高度相似，softmax 雅可比近于抵消，")
    print("    导致注意力打分参数 W_q 的梯度天然极小 —— 比 loss 小 8 个数量级。")
    print("    此时逐元素相对误差必然很大，但公式是正确的。判据必须区分这两件事。）")

    enc, dec = make(seed=0)
    src, mask, tgt_in = sample(B=2, S=6, T=5, seed=3)
    hs, hf = enc.forward(src, mask)
    G = np.random.default_rng(3).normal(size=(2, 5, V_TGT))

    def loss_fn():
        lg, _ = dec.forward(tgt_in, hs, mask, hf)
        return float((lg * G).sum())

    loss_fn()
    dec.zero_grad()
    dec.backward(G)
    res = check_gradient(dec.named_parameters_dict(), dec.named_grads(), loss_fn,
                         n_samples=4, eps=1e-5, rng=np.random.default_rng(0))
    by_name = {r["name"]: r for r in res}

    check("检验不因小梯度而误报 FAIL", all(r["passed"] for r in res),
          str([r["name"] for r in res if not r["passed"]]))
    check("attn.W_q 被标记为 low-snr", by_name["attn.W_q"]["status"] == "low-snr",
          by_name["attn.W_q"]["status"])
    check("attn.W_q 的相对误差读数无意义（nan）",
          by_name["attn.W_q"]["max_rel_err"] != by_name["attn.W_q"]["max_rel_err"])
    check("attn.W_q 的分歧/抖动仍是 O(1)",
          by_name["attn.W_q"]["ratio"] < 1e3, f"{by_name['attn.W_q']['ratio']:.2e}")
    check("梯度健康的张量仍有个位数量级的比值",
          by_name["out.b"]["ratio"] < 1e3 and by_name["out.b"]["status"] == "ok",
          f"out.b ratio={by_name['out.b']['ratio']:.2e}")

    # 决定性证据：数值估计随 ε 增大收敛到解析值
    name = "attn.W_q"
    par = dec.named_parameters_dict()[name]
    flat_g = np.ascontiguousarray(dec.named_grads()[name]).reshape(-1)
    j = int(np.argmax(np.abs(flat_g)))
    flat_p = par.reshape(-1)
    ests = []
    for eps in (1e-6, 1e-5, 1e-4, 1e-3):
        old = flat_p[j]
        flat_p[j] = old + eps; fp = loss_fn()
        flat_p[j] = old - eps; fm = loss_fn()
        flat_p[j] = old
        ests.append((eps, (fp - fm) / (2 * eps)))
    ana = float(flat_g[j])
    rels = [abs(v - ana) / max(abs(v) + abs(ana), 1e-30) for _, v in ests]
    check("数值估计随 ε 增大收敛到解析值", rels[-1] < rels[0],
          " -> ".join(f"{r:.1e}" for r in rels))
    check("最细步长下相对差已降到 1e-4 以内", rels[-1] < 1e-3, f"{rels[-1]:.2e}")


def test_real_data() -> None:
    print("\n== 真实数据端到端 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(4)))
    enc = GRUEncoder(len(train.src_vocab), CFG.src_emb_dim, CFG.hidden_dim,
                     seed=0, pad_id=train.src_vocab.pad_id)
    dec = AttentionDecoder(len(train.tgt_vocab), CFG.tgt_emb_dim, CFG.hidden_dim,
                           CFG.hidden_dim, CFG, seed=0, pad_id=train.tgt_vocab.pad_id)

    hs, hf = enc.forward(batch.src, batch.src_mask)
    logits, weights = dec.forward(batch.tgt_in, hs, batch.src_mask, hf)
    check("logits 形状符合词表", logits.shape == (4, batch.tgt_in.shape[1], len(train.tgt_vocab)),
          str(logits.shape))
    check("权重形状 (B,T,S)", weights.shape == (4, batch.tgt_in.shape[1], batch.src.shape[1]))
    check("输出无 nan", bool(np.isfinite(logits).all() and np.isfinite(weights).all()))
    check("权重行和为 1", close(weights.sum(-1), 1.0, tol=1e-12))
    padded = weights[np.broadcast_to(batch.src_mask[:, None, :] < 0.5, weights.shape)]
    check("padding 权重严格为 0", bool(np.all(padded == 0.0)))

    # 整链路梯度检验（小配置，控制耗时）
    small_enc = GRUEncoder(len(train.src_vocab), 8, 6, seed=1, pad_id=train.src_vocab.pad_id)
    small_dec = AttentionDecoder(len(train.tgt_vocab), 8, 6, 6, CFG, seed=1,
                                 pad_id=train.tgt_vocab.pad_id)
    s_src = batch.src[:2, :6]
    s_mask = batch.src_mask[:2, :6]
    s_tgt = batch.tgt_in[:2, :5]
    G = np.random.default_rng(0).normal(size=(2, 5, len(train.tgt_vocab)))

    def loss_fn():
        hs2, hf2 = small_enc.forward(s_src, s_mask)
        lg2, _ = small_dec.forward(s_tgt, hs2, s_mask, hf2)
        return float((lg2 * G).sum())

    def ana():
        small_enc.zero_grad(); small_dec.zero_grad()
        hs2, hf2 = small_enc.forward(s_src, s_mask)
        lg2, _ = small_dec.forward(s_tgt, hs2, s_mask, hf2)
        dd = small_dec.backward(G)
        small_enc.backward(dd["src_h"], dd["src_h_final"])
        out = {f"enc.{k}": v for k, v in small_enc.named_grads().items()}
        out.update({f"dec.{k}": v for k, v in small_dec.named_grads().items()})
        return out

    loss_fn()
    a = ana()
    tensors = {f"enc.{k}": v for k, v in small_enc.named_parameters_dict().items()}
    tensors.update({f"dec.{k}": v for k, v in small_dec.named_parameters_dict().items()})
    res = check_gradient(tensors, a, loss_fn, n_samples=3, eps=1e-5,
                         rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("真实词表下整链路梯度检验通过", s["passed"], f"最大分歧/抖动 {s['worst_ratio']:.2e}")


def main() -> None:
    print("=" * 70)
    print(" 解码器（带注意力）测试")
    print("=" * 70)
    test_shapes()
    test_teacher_forcing_and_causality()
    test_output_and_init_modes()
    test_errors()
    test_decoder_gradients()
    test_end_to_end_chain()
    test_missing_value_grad_regression()
    test_gradcheck_criterion()
    test_decoder_low_snr_regression()
    test_real_data()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
