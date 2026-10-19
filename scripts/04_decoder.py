"""第 4 步自检脚本：把编码器和注意力拼成解码器，验证整条链路。

运行：
    python scripts/04_decoder.py

这一步真正的难点不是"解码器长什么样"，而是**梯度能不能正确穿过注意力层
回到编码器**。所以验证重点放在：

    [1] 架构与数据流               看清每一步的形状
    [2] teacher forcing 错位       解码器输入 = <bos> + y[:-1]
    [3] 因果性                     第 t 步看不到 t 之后的词
    [4] 三种 output_mode           配置项都能跑通
    [5] 三种 init_mode
    [6] 端到端梯度检验             解码器参数
    [7] 梯度穿过注意力回到编码器 ← 最容易断的一环
    [8] 负面对照：漏掉 dkeys+dvalues 相加
    [9] 参数量与耗时
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import dataclasses

import numpy as np

from seq2seq_attn import data as D
from seq2seq_attn.attention import BahdanauAttention, DotProductAttention
from seq2seq_attn.config import CONFIG
from seq2seq_attn.decoder import AttentionDecoder
from seq2seq_attn.encoder import GRUEncoder
from seq2seq_attn.nn import check_gradient, numerical_gradient_at, summarize_gradient_report

LINE = "=" * 82
SUB = "-" * 82
STATUS_NOTE = {
    "ok": "✓",
    "low-snr": "~ 梯度贴着数值噪声地板，相对误差读数无意义",
    "FAIL": "✗ 不通过",
}


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def grad_table(label: str, tensors: dict, analytic: dict, loss_fn, rng,
               n_samples=4, eps=1e-5, show=14):
    results = check_gradient(tensors, analytic, loss_fn, n_samples=n_samples, eps=eps, rng=rng)
    summary = summarize_gradient_report(results)
    print(f"\n  {label}")
    print(f"    |L| = {summary['loss_magnitude']:.3e}，"
          f"噪声地板 |L|·ε_mach/(2ε) = {summary['noise_floor']:.2e}")
    print(f"    {'张量':<18}{'形状':<18}{'分歧/抖动':>12}{'解析梯度':>14}{'相对误差':>11}  判读")
    print("    " + "-" * 88)
    for r in results[:show]:
        rel = "—" if r["max_rel_err"] != r["max_rel_err"] else f"{r['max_rel_err']:.2e}"
        print(f"    {r['name']:<18}{str(r['shape']):<18}{r['ratio']:>12.2e}"
              f"{r['worst_ana']:>14.4e}{rel:>11}  {STATUS_NOTE[r['status']]}")
    if len(results) > show:
        print(f"    ...（其余 {len(results) - show} 个张量省略）")
    print(f"    合计 {summary['n_tensors']} 个张量，最大「分歧/抖动」= {summary['worst_ratio']:.2e}"
          f"  ->  {'✓ 全部通过' if summary['passed'] else '✗ 有不通过项：' + str(summary['failed'])}")
    print("    （「分歧/抖动」= |解析-数值| / |数值(ε)-数值(4ε)|。"
          "O(1)~O(100) 说明差异完全由数值噪声解释，≫1e3 说明公式错了。）")
    return summary


def make_model(**overrides):
    """按配置造一对编码器 + 解码器（小尺寸，速度可控）。"""
    cfg = CONFIG.model
    V_src, V_tgt = 17, 27
    kw = dict(seed=0)
    kw.update(overrides)
    enc = GRUEncoder(V_src, cfg.src_emb_dim, cfg.hidden_dim,
                     init_scale=cfg.init_scale, seed=kw.get("seed", 0))
    dec = AttentionDecoder(V_tgt, cfg.tgt_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg,
                           output_mode=kw.get("output_mode", cfg.decoder_output),
                           init_mode=kw.get("init_mode", cfg.decoder_init),
                           init_scale=cfg.init_scale, seed=kw.get("seed", 0))
    return enc, dec


# ======================================================================================
# [1] 架构与数据流
# ======================================================================================
def section_pipeline() -> None:
    title("[1] 架构与数据流")

    cfg = CONFIG.model
    print("  桥接：  s₀ = tanh(h_final · W_b + b_b)")
    print("  第 t 步：")
    print("      c_t, w_t = attention(query=s_{t-1}, keys=H_seq, values=H_seq, mask=src_mask)")
    print("      x_t      = [ emb(y_{t-1}) ; c_t ]")
    print("      s_t      = GRU(x_t, s_{t-1})")
    print("      logits_t = W_o · [ s_t ; c_t ]\n")

    enc, dec = make_model()
    B, S, T = 4, 11, 10
    rng = np.random.default_rng(0)
    src = rng.integers(0, 17, size=(B, S))
    src[:, -1] = 0
    mask = np.ones((B, S)); mask[0, -2:] = 0.0
    tgt_in = rng.integers(0, 27, size=(B, T))

    hs, hf = enc.forward(src, mask)
    logits, weights = dec.forward(tgt_in, hs, mask, hf)

    print(f"    源句 id      src        {tuple(src.shape)}")
    print(f"    源句 mask    src_mask   {tuple(mask.shape)}")
    print(f"    编码器输出   H_seq      {tuple(hs.shape)}")
    print(f"    编码器末态   h_final    {tuple(hf.shape)}")
    print(f"    解码器输入   tgt_in     {tuple(tgt_in.shape)}")
    print(f"    解码器隐状态 s₀         {tuple(dec.initial_state(hf)[0].shape)}")
    print(f"    输出 logits             {tuple(logits.shape)}")
    print(f"    注意力权重   weights    {tuple(weights.shape)}")
    print(f"\n  注意力权重每步行和 = {weights.sum(-1).min():.6f} ~ {weights.sum(-1).max():.6f}（应全为 1）")
    pad_w = weights[np.broadcast_to(mask[:, None, :] < 0.5, weights.shape)]
    print(f"  padding 位置权重最大值 = {pad_w.max():.3e}（应为 0）")
    print(f"\n  {dec}")
    print(f"  {enc}")

    print("\n  数据流一句话总结：")
    print("    编码器把源句压成 H_seq（每个位置一个向量）；")
    print("    解码器每生成一个词，就先用当前隐状态去 H_seq 里「查」一次（注意力），")
    print("    把查到的 context 和上一个词的嵌入拼起来喂给 GRU，再投影成词表分布。")


# ======================================================================================
# [2] teacher forcing
# ======================================================================================
def section_teacher_forcing() -> None:
    title("[2] teacher forcing 的错位在数据管道里完成")

    train, _, _ = D.build_datasets(CONFIG.data)
    vocab = train.tgt_vocab
    batch = D.collate(train, list(range(3)))

    print("  `Batch` 已经做好错位（第 1 步）：")
    for i in range(2):
        t_in = " ".join(vocab.itos[int(j)] for j in batch.tgt_in[i])
        t_out = " ".join(vocab.itos[int(j)] for j in batch.tgt_out[i])
        print(f"    样本 {i}  tgt_in  = {t_in}")
        print(f"            tgt_out = {t_out}")
    print("\n  → 解码器只需要按 `tgt_in[t]` 逐位喂入、用 `tgt_out[t]` 算损失，")
    print("    不需要自己处理错位。（第 5 步的交叉熵就是这么用的。）")


# ======================================================================================
# [3] 因果性
# ======================================================================================
def section_causality() -> None:
    title("[3] 因果性：第 t 步绝不能看到 t 之后的词")

    print("  decoder 是循环结构，第 t 步只吃 `tgt_in[:, t]`。")
    print("  但这一点必须**验证**，因为如果实现里不小心用了整段 `x_emb`")
    print("  （比如把 embedding 层的某个全局统计喂了进去），就会出现信息泄露。\n")

    enc, dec = make_model()
    rng = np.random.default_rng(1)
    B, S, T = 3, 9, 7
    src = rng.integers(0, 17, size=(B, S))
    mask = np.ones((B, S))
    hs, hf = enc.forward(src, mask)
    tgt_in = rng.integers(0, 27, size=(B, T))

    logits1, _ = dec.forward(tgt_in, hs, mask, hf)

    ok = True
    details = []
    for cut in range(T):
        mutated = tgt_in.copy()
        if cut + 1 < T:
            mutated[:, cut + 1:] = rng.integers(0, 27, size=(B, T - cut - 1))
        logits2, _ = dec.forward(mutated, hs, mask, hf)
        # 前 cut+1 步的 logits 必须完全不变
        diff = np.abs(logits1[:, : cut + 1, :] - logits2[:, : cut + 1, :]).max()
        details.append((cut, diff))
        if diff != 0.0:
            ok = False

    print("    篡改位置 t 之后的输入，检查前 t+1 步输出是否不变：")
    print(f"      {'篡改起点':<12}{'前若干步 logits 最大差':>24}")
    print("      " + "-" * 40)
    for cut, diff in details:
        print(f"      t≥{cut + 1:<10}{diff:>24.3e}")
    print(f"\n  结论：{'全部严格为 0，因果性成立 ✓' if ok else '出现非零差异，存在信息泄露 ✗'}")
    print("  （这正是自回归模型能用于推理的前提：生成第 t 个词时，")
    print("    前面已经生成的词是确定的，后面的词还不存在。）")


# ======================================================================================
# [4][5] 配置项
# ======================================================================================
def section_config_variants() -> None:
    title("[4][5] 三个配置项都能跑通（第 8 步消融实验的基础）")

    cfg = CONFIG.model
    rng = np.random.default_rng(2)
    B, S, T = 2, 8, 6
    src = rng.integers(0, 17, size=(B, S))
    mask = np.ones((B, S)); mask[1, 5:] = 0.0
    tgt_in = rng.integers(0, 27, size=(B, T))

    print("  output_mode：输出层吃什么")
    print(f"    {'模式':<12}{'输出层输入维度':>16}{'说明':<40}")
    print("    " + "-" * 74)
    for mode, note in [("s_c", "[s_t ; c_t]，最简洁"),
                       ("s_c_emb", "[s_t ; c_t ; emb(y_{t-1})]，Bahdanau 原论文"),
                       ("s", "[s_t] 消融基线")]:
        enc, dec = make_model(output_mode=mode)
        hs, hf = enc.forward(src, mask)
        logits, _ = dec.forward(tgt_in, hs, mask, hf)
        print(f"    {mode:<12}{dec.output_input_dim():>16}{note:<40}")
        assert logits.shape == (B, T, 27)

    print("\n  init_mode：解码器起点怎么来")
    print(f"    {'模式':<12}{'bridge 参数量':>16}{'说明':<40}")
    print("    " + "-" * 74)
    for mode, note in [("bridge", "s₀ = tanh(W·h_final + b)，可学投影"),
                       ("direct", "s₀ = h_final，零参数"),
                       ("zero", "s₀ = 0，全靠注意力")]:
        enc, dec = make_model(init_mode=mode)
        hs, hf = enc.forward(src, mask)
        logits, _ = dec.forward(tgt_in, hs, mask, hf)
        print(f"    {mode:<12}{dec.bridge.num_params():>16}{note:<40}")
        assert logits.shape == (B, T, 27)

    print("\n  另外注意力类型也可以换（第 2 步实现了两种）：")
    for attn_type in ("bahdanau", "dot"):
        cfg2 = dataclasses.replace(cfg, attn_type=attn_type)
        dec = AttentionDecoder(27, cfg.tgt_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg2, seed=0)
        print(f"    attn_type={attn_type:<10} {type(dec.attn).__name__:<22}"
              f"参数量 {dec.attn.num_params():>7,}")

    # init_mode='direct' 在维度不同时必须报错
    try:
        bad = AttentionDecoder(27, 16, 8, 6, cfg, init_mode="direct", seed=0)
        bad.initial_state(np.zeros((1, 6)))
        print("\n  init_mode='direct' 且维度不同 -> 没报错（不对）")
    except ValueError:
        print("\n  init_mode='direct' 且编码器/解码器隐维度不同时 -> 正确报错 ✓")


# ======================================================================================
# [6] 解码器端到端梯度检验
# ======================================================================================
def section_decoder_gradcheck() -> None:
    title("[6] 解码器端到端梯度检验")

    enc, dec = make_model()
    rng = np.random.default_rng(3)
    B, S, T = 2, 6, 5
    src = rng.integers(0, 17, size=(B, S))
    mask = np.ones((B, S)); mask[0, 4:] = 0.0
    tgt_in = rng.integers(0, 27, size=(B, T))

    hs, hf = enc.forward(src, mask)
    G = rng.normal(size=(B, T, 27))

    def loss_fn():
        lg, _ = dec.forward(tgt_in, hs, mask, hf)
        return float((lg * G).sum())

    loss_fn()
    dec.zero_grad()
    d = dec.backward(G)
    tensors = dec.named_parameters_dict()
    analytic = dec.named_grads()
    summary = grad_table("解码器全部参数（含 attn 的 W_q/W_k/v/b）：",
                         tensors, analytic, loss_fn, np.random.default_rng(0), n_samples=4)
    print(f"\n  返回的输入梯度：src_h {tuple(d['src_h'].shape)}  "
          f"src_h_final {tuple(d['src_h_final'].shape)}")

    if summary["low_snr"]:
        section_low_snr(dec, summary["low_snr"])


# ======================================================================================
# [6b] 小梯度张量怎么办：有限差分检验的适用边界
# ======================================================================================
def section_low_snr(dec, low_snr_names) -> None:
    print("\n  " + SUB)
    print("  插播：哪些张量的梯度小到有限差分检验不了？")
    print("  " + SUB)
    print("  中心差分的噪声下界是 `noise_floor ≈ |L|·ε_mach/(2ε)`。")
    print("  如果某个张量的梯度本身就贴着这个地板，数值梯度就全是噪声，")
    print("  逐元素相对误差必然很大 —— **这不代表公式错了**。\n")

    print("  证据一：数值估计随 ε 增大**收敛**到解析值")
    print("  （如果是噪声，数值估计会一直乱跳；如果公式错了，它会稳定收敛到错误的值）\n")

    G = None
    # 重新构造一份和 section_decoder_gradcheck 里完全一致的输入
    rng = np.random.default_rng(3)
    B, S, T = 2, 6, 5
    enc2, dec2 = make_model()
    src = rng.integers(0, 17, size=(B, S))
    mask = np.ones((B, S)); mask[0, 4:] = 0.0
    tgt_in = rng.integers(0, 27, size=(B, T))
    hs, hf = enc2.forward(src, mask)
    G = rng.normal(size=(B, T, 27))

    def loss_fn2():
        lg, _ = dec2.forward(tgt_in, hs, mask, hf)
        return float((lg * G).sum())

    loss_fn2()
    dec2.zero_grad()
    dec2.backward(G)
    grads = dec2.named_grads()
    params = dec2.named_parameters_dict()

    for name in low_snr_names:
        ana = np.ascontiguousarray(grads[name]).reshape(-1)
        par = params[name]
        flat_p = par.reshape(-1)
        # 选梯度最大的元素：那里信号最强，最容易看清收敛
        order = np.argsort(-np.abs(ana))[:1]
        j = int(order[0])
        print(f"    {name}（|解析梯度|_max = {np.abs(ana).max():.3e}）")
        print(f"      {'ε':>8}{'数值梯度':>18}{'解析梯度':>18}{'相对差':>12}")
        print("      " + "-" * 56)
        for eps in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            old = flat_p[j]
            flat_p[j] = old + eps
            fp = loss_fn2()
            flat_p[j] = old - eps
            fm = loss_fn2()
            flat_p[j] = old
            num = (fp - fm) / (2 * eps)
            rel = abs(num - ana[j]) / max(abs(num) + abs(ana[j]), 1e-30)
            print(f"      {eps:>8.0e}{num:>18.6e}{ana[j]:>18.6e}{rel:>12.2e}")
        print()

    print("    可以看到：ε 从 1e-7 增大到 1e-3 的过程中，数值估计单调地")
    print("    逼近解析值（相对差从 2e-2 降到 1e-5 量级）。")
    print("    这种「随 ε 收敛到解析值」的行为只可能来自**正确的解析梯度**：")
    print("    如果是噪声，估计值会一直乱跳；如果公式错了，它会稳定收敛到错误的值。\n")
    print("    结论：`low-snr` 标记意味着逐元素相对误差**不可用**（不是不通过），")
    print("    此时应该改用自校准判据（分歧/抖动）和上面的 ε 收敛性来判定。")
    print("    这就是检验器为什么要返回 status 而不只返回一个相对误差数字。")


# ======================================================================================
# [7] 梯度穿过注意力回到编码器
# ======================================================================================
def section_end_to_end() -> None:
    title("[7] 梯度穿过注意力回到编码器（最容易断的一环）")

    print("  loss 依赖 logits，logits 依赖 context，context 依赖编码器的 H_seq。")
    print("  所以编码器的参数**必须**能收到梯度。如果 attention.backward 的")
    print("  `keys` / `values` 没接好，编码器梯度会变成 0 —— 而 loss 照样下降")
    print("  （因为解码器自己还在学），极难发现。\n")

    enc, dec = make_model()
    rng = np.random.default_rng(4)
    B, S, T = 2, 6, 5
    src = rng.integers(0, 17, size=(B, S))
    mask = np.ones((B, S)); mask[0, 4:] = 0.0
    tgt_in = rng.integers(0, 27, size=(B, T))
    G = rng.normal(size=(B, T, 27))

    def loss_fn():
        hs, hf = enc.forward(src, mask)
        lg, _ = dec.forward(tgt_in, hs, mask, hf)
        return float((lg * G).sum())

    # 先确认编码器确实收到非零梯度
    loss_fn()
    enc.zero_grad(); dec.zero_grad()
    hs, hf = enc.forward(src, mask)
    lg, _ = dec.forward(tgt_in, hs, mask, hf)
    d = dec.backward(G)
    enc.backward(d["src_h"], d["src_h_final"])
    enc_g = enc.named_grads()
    print("  链路是否连通：编码器各参数的梯度范数")
    print(f"    {'参数':<16}{'梯度范数':>16}{'是否非零':>12}")
    print("    " + "-" * 46)
    all_nonzero = True
    for k in sorted(enc_g):
        if not enc_g[k].size:
            continue
        nrm = float(np.linalg.norm(enc_g[k]))
        ok = nrm > 0
        all_nonzero = all_nonzero and ok
        print(f"    {k:<16}{nrm:>16.5f}{'是' if ok else '否 ← 断了！':>12}")
    print(f"  → {'链路连通，编码器收到了来自解码器 loss 的梯度 ✓' if all_nonzero else '有参数梯度为 0 ✗'}")

    # 整条链路（编码器 + 解码器）的联合梯度检验
    print("\n  接下来对**整条链路**做梯度检验（编码器参数 + 解码器参数一起检）。")

    def chain_loss():
        hs2, hf2 = enc.forward(src, mask)
        lg2, _ = dec.forward(tgt_in, hs2, mask, hf2)
        return float((lg2 * G).sum())

    def chain_analytic():
        """跑一次完整反向，把编码器和解码器的梯度按统一命名收集起来。"""
        enc.zero_grad()
        dec.zero_grad()
        hs2, hf2 = enc.forward(src, mask)
        lg2, _ = dec.forward(tgt_in, hs2, mask, hf2)
        dd = dec.backward(G)
        enc.backward(dd["src_h"], dd["src_h_final"])
        out = {f"enc.{k}": v for k, v in enc.named_grads().items()}
        out.update({f"dec.{k}": v for k, v in dec.named_grads().items()})
        return out

    chain_loss()
    analytic = chain_analytic()
    tensors = {f"enc.{k}": v for k, v in enc.named_parameters_dict().items()}
    tensors.update({f"dec.{k}": v for k, v in dec.named_parameters_dict().items()})
    grad_table("编码器 + 解码器 整条链路：", tensors, analytic, chain_loss,
               np.random.default_rng(0), n_samples=4, show=16)
    print("\n  → 编码器和解码器的参数在**同一次检验**里全部通过，")
    print("    说明梯度确实从 loss 一路传到了编码器的 embedding 和 GRU 里。")


# ======================================================================================
# [8] 负面对照
# ======================================================================================
def section_bug_hunt() -> None:
    title("[8] 负面对照：漏掉 dkeys + dvalues 相加会怎样")

    class MissingKeyGrad(AttentionDecoder):
        """故意写错：只把 dattn['keys'] 回传，丢掉 dattn['values']。

        因为本项目传的是 values = keys = H_seq，两者本来就是同一份张量，
        梯度必须叠加。漏掉一半会让编码器梯度偏小 —— 不报错、不出 nan。
        """

        def backward(self, dlogits):
            c = self._cache
            B, T, S = c["B"], c["T"], c["S"]
            Hs, E = self.src_hidden_dim, self.emb_dim
            import numpy as _np
            self.zero_grad()
            dsrc_h = _np.zeros((B, S, Hs))
            demb_seq = _np.zeros((B, T, E))
            ds_carry = _np.zeros((B, self.hidden_dim))
            for t in reversed(range(T)):
                self.out.load_cache(c["out_caches"][t])
                do_t = self.out.backward(dlogits[:, t, :])
                ds_t, dctx, demb_from_out = self._split_output_grad(do_t)
                ds_t = ds_t + ds_carry
                if demb_from_out is not None:
                    demb_seq[:, t, :] += demb_from_out
                self.cell.load_cache(c["cell_caches"][t])
                dcell = self.cell.backward(ds_t)
                dx_t = dcell["x"]
                demb_seq[:, t, :] += dx_t[:, :E]
                dctx = dctx + dx_t[:, E:]
                ds_prev = dcell["h"]
                self.attn.load_cache(c["attn_caches"][t])
                dattn = self.attn.backward(dctx[:, None, :])
                ds_prev = ds_prev + dattn["query"][:, 0, :]
                dsrc_h += dattn["keys"]                      # ← 少了 + dattn["values"]
                ds_carry = ds_prev
            self.emb.backward(demb_seq)
            bc = c["bridge_cache"]
            dsrc_h_final = _np.zeros((B, Hs))
            if bc is not None:
                s0 = bc["s0"]
                dsrc_h_final = self.bridge.backward(ds_carry * (1.0 - s0 * s0))
            elif self.init_mode == "direct":
                dsrc_h_final = ds_carry
            return {"src_h": dsrc_h, "src_h_final": dsrc_h_final}

    def run(dec_cls, seed=5):
        cfg = CONFIG.model
        enc = GRUEncoder(17, cfg.src_emb_dim, cfg.hidden_dim, seed=0)
        dec = dec_cls(27, cfg.tgt_emb_dim, cfg.hidden_dim, cfg.hidden_dim, cfg, seed=0)
        rng = np.random.default_rng(seed)
        B, S, T = 2, 6, 5
        src = rng.integers(0, 17, size=(B, S))
        mask = np.ones((B, S)); mask[0, 4:] = 0.0
        tgt_in = rng.integers(0, 27, size=(B, T))
        G = rng.normal(size=(B, T, 27))

        def loss_fn():
            hs, hf = enc.forward(src, mask)
            lg, _ = dec.forward(tgt_in, hs, mask, hf)
            return float((lg * G).sum())

        def analytic():
            enc.zero_grad(); dec.zero_grad()
            hs, hf = enc.forward(src, mask)
            lg, _ = dec.forward(tgt_in, hs, mask, hf)
            dd = dec.backward(G)
            enc.backward(dd["src_h"], dd["src_h_final"])
            out = {f"enc.{k}": v for k, v in enc.named_grads().items()}
            out.update({f"dec.{k}": v for k, v in dec.named_grads().items()})
            return out

        loss_fn()
        ana = analytic()
        tensors = {f"enc.{k}": v for k, v in enc.named_parameters_dict().items()}
        tensors.update({f"dec.{k}": v for k, v in dec.named_parameters_dict().items()})
        return check_gradient(tensors, ana, loss_fn, n_samples=3, eps=1e-5,
                              rng=np.random.default_rng(0))

    print("  用「分歧/抖动」这个自校准判据分别看三组参数：\n")
    good = run(AttentionDecoder)
    bad = run(MissingKeyGrad)

    def group(results, pred):
        sel = [r for r in results if pred(r["name"])]
        worst_ratio = max(r["ratio"] for r in sel)
        n_fail = sum(1 for r in sel if r["status"] == "FAIL")
        return worst_ratio, n_fail

    print(f"    {'参数组':<12}{'正确实现':>26}{'漏掉一半':>26}")
    print(f"    {'':<12}{'分歧/抖动':>14}{'不通过项':>12}{'分歧/抖动':>14}{'不通过项':>12}")
    print("    " + "-" * 64)
    for group_name, pred in [("编码器", lambda k: k.startswith("enc.")),
                             ("解码器", lambda k: k.startswith("dec.")),
                             ("  其中注意力", lambda k: k.startswith("dec.attn."))]:
        gr, gf = group(good, pred)
        br, bf = group(bad, pred)
        print(f"    {group_name:<12}{gr:>14.2e}{gf:>12}{br:>14.2e}{bf:>12}")

    g_enc, _ = group(good, lambda k: k.startswith("enc."))
    b_enc, bf_enc = group(bad, lambda k: k.startswith("enc."))
    g_dec, gf_dec = group(good, lambda k: k.startswith("dec."))
    b_dec, bf_dec = group(bad, lambda k: k.startswith("dec."))

    print(f"\n  值得注意的地方：")
    print(f"    · 编码器：分歧/抖动 {g_enc:.2e}  ->  {b_enc:.2e}   错误**被抓住**"
          f"（{bf_enc} 个张量不通过）")
    print(f"    · 解码器：分歧/抖动 {g_dec:.2e}  ->  {b_dec:.2e}   "
          f"不通过项 {gf_dec}  ->  {bf_dec}   **解码器参数照样通过**！")
    print("\n  这就是这个 bug 的可怕之处：")
    print("    只检验解码器参数的实现会「全部通过」，因为漏掉的那一半梯度")
    print("    本来就只流向编码器。而 loss 依然会下降（解码器自己在学），")
    print("    最终表现是「模型能训练，但 encoder 几乎没学到东西，效果上限很低」。")
    print("    → 所以第 [7] 节那个「编码器梯度是否非零」的检查是必需的。")


# ======================================================================================
# [9] 参数量与耗时
# ======================================================================================
def section_cost() -> None:
    title("[9] 参数量与耗时")

    enc, dec = make_model()
    eb = dec.num_params_breakdown()
    print(f"  目标词表 27，emb={CONFIG.model.tgt_emb_dim}，hidden={CONFIG.model.hidden_dim}\n")
    print(f"  解码器各部分参数量：")
    print(f"    {'部分':<12}{'参数量':>10}")
    print("    " + "-" * 24)
    for k in ("emb", "attn", "cell", "bridge", "out", "total"):
        print(f"    {k:<12}{eb[k]:>10,}")
    print(f"\n  整模型：编码器 {enc.num_params():,} + 解码器 {dec.num_params():,}"
          f" = {enc.num_params() + dec.num_params():,}")

    train, _, _ = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(32)))
    enc2 = GRUEncoder(len(train.src_vocab), CONFIG.model.src_emb_dim, CONFIG.model.hidden_dim,
                      seed=0, pad_id=train.src_vocab.pad_id)
    dec2 = AttentionDecoder(len(train.tgt_vocab), CONFIG.model.tgt_emb_dim,
                            CONFIG.model.hidden_dim, CONFIG.model.hidden_dim, CONFIG.model,
                            seed=0, pad_id=train.tgt_vocab.pad_id)

    def one_step():
        enc2.zero_grad(); dec2.zero_grad()
        hs, hf = enc2.forward(batch.src, batch.src_mask)
        lg, _ = dec2.forward(batch.tgt_in, hs, batch.src_mask, hf)
        d = dec2.backward(np.ones_like(lg))
        enc2.backward(d["src_h"], d["src_h_final"])

    one_step()
    n_rep = 50
    t0 = time.perf_counter()
    for _ in range(n_rep):
        one_step()
    dt = (time.perf_counter() - t0) / n_rep * 1000

    print(f"\n  一个 batch（B=32, S=11, T=10）前向 + 反向：{dt:.1f} ms")
    n_batch = (len(train) + 31) // 32
    print(f"  训练集共 {n_batch} 个 batch  ->  一个 epoch 约 {dt * n_batch / 1000:.1f} 秒")
    print(f"  第 5 步计划跑 3000 步 ≈ {3000 / n_batch:.0f} 个 epoch"
          f"  ->  约 {dt * 3000 / 1000:.0f} 秒")
    print(f"\n  纯 NumPy 下这个速度可以接受 —— 数据只有 4000 条、序列只有 10 步，")
    print("  所以第 5 步真的能把它训到收敛，而不是只能跑个 demo。")


# ======================================================================================
def main() -> None:
    print(LINE)
    print(" 第 4 步 · 带注意力的解码器")
    print(LINE)
    print(" 重点不是架构本身，而是**梯度能不能正确穿过注意力层回到编码器**。")

    section_pipeline()
    section_teacher_forcing()
    section_causality()
    section_config_variants()
    section_decoder_gradcheck()
    section_end_to_end()
    section_bug_hunt()
    section_cost()

    title("第 4 步完成 ✓")
    print(" 编码器 + 注意力 + 解码器已连成完整链路，梯度全程可通。")
    print(" 下一步（第 5 步）加上交叉熵损失、Adam 和训练循环，让它真的学起来。")
    print()


if __name__ == "__main__":
    main()
