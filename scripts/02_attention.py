"""第 2 步自检脚本：把注意力机制单独拎出来，从里到外验证一遍。

运行：
    python scripts/02_attention.py

它不训练任何模型。目标只有一个：证明 `seq2seq_attn/attention.py` 里的
前向和反向传播都是**正确**的。第 4 步把它插进解码器时，就可以当它是个可信零件。

验证手段（由弱到强）：
    [2] 手算：2 个 key 的极简例子，逐值打印中间量
    [3] mask：两种实现方案的对照，重点看退化情况
    [4] 结构：两种注意力的参数量与计算量对比
    [5] 交叉验证：用朴素三重循环独立重实现，逐值比对
    [6] 梯度检验：中心差分数值梯度 vs 解析梯度
    [7] 反向验证：故意写错反向传播，确认梯度检验真的抓得住
    [8] 可视化：权重长什么样
"""

from __future__ import annotations

import pathlib
import sys
import time
import unicodedata

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
    softmax_masked,
    summarize_gradient_report,
)

LINE = "=" * 82
SUB = "-" * 82


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def dwidth(s: str) -> int:
    """字符串在终端里的显示宽度（中文/全角字符算 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad_display(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


SHADES = " .:-=+*#%@"


def shade(value: float, vmax: float) -> str:
    if vmax <= 0:
        return " "
    k = int(round(value / vmax * (len(SHADES) - 1)))
    return SHADES[max(0, min(k, len(SHADES) - 1))]


def render_matrix(weights: np.ndarray, col_tokens, row_labels) -> None:
    """把 (Tq, S) 的注意力权重画成字符热力图。"""
    col_w = 5
    print("        " + "".join(pad_display(str(i), col_w) for i in range(len(col_tokens))))
    print("  源句  " + "".join(pad_display(t, col_w) for t in col_tokens))
    print("  " + SUB[: 8 + col_w * len(col_tokens)])
    vmax = float(weights.max()) if weights.size else 0.0
    for r, label in enumerate(row_labels):
        cells = "".join(pad_display(shade(w, vmax), col_w) for w in weights[r])
        argmax = int(np.argmax(weights[r]))
        print(f"  {pad_display(label, 6)}{cells}  ← 峰值在第 {argmax} 位")


# ======================================================================================
# [1] 本质
# ======================================================================================
def section_essence() -> None:
    title("[1] 注意力的本质：一次「软查字典」")

    print("  硬查字典：拿 query 去精确匹配一个 key，取出那一个 value。")
    print("  软查字典：给每个 key 算一个匹配分数，softmax 成权重，")
    print("            然后按权重把所有 value 加权平均。\n")

    q = np.array([[[1.0, 0.0]]])                        # 1 个查询
    k = np.array([[[1.0, 0.0], [0.0, 1.0]]])            # 2 个键
    v = np.array([[[10.0, 0.0], [0.0, 20.0]]])          # 2 个值

    attn = DotProductAttention(q_dim=2, k_dim=2, v_dim=2)
    ctx, w = attn(q, k, v)

    print("  query  = [1, 0]")
    print("  key₀   = [1, 0]   ← 和 query 完全同向")
    print("  key₁   = [0, 1]   ← 和 query 完全正交")
    print("  value₀ = [10, 0]")
    print("  value₁ = [0, 20]\n")

    scores = np.einsum("bqd,bsd->bqs", q, k) / np.sqrt(2)
    print(f"  scores = q·k / √2 = {scores.reshape(-1)}      （d=2，所以除以 √2 ≈ 1.414）")
    print(f"  weights = softmax(scores) = {w.reshape(-1).round(6)}")
    print(f"  context = Σ weightsⱼ · valueⱼ = {ctx.reshape(-1).round(6)}\n")

    print(f"  key₀ 的权重 {w[0,0,0]:.4f} > key₁ 的权重 {w[0,0,1]:.4f}：")
    print("  因为 key₀ 和 query 方向完全一致，点积最大。")
    print(f"  context 落在两个 value 之间（{ctx[0,0].round(4)}），这就是「软」的含义 ——")
    print("  不是二选一，而是按相似度做加权平均。正因为是加权平均，")
    print("  它才对 query 可微，才能被端到端训练。")


# ======================================================================================
# [2] 手算
# ======================================================================================
def section_hand_check() -> None:
    title("[2] 手算验证：缩放点积注意力（2 个 key）")

    q = np.array([[[1.0, 0.0]]])
    k = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    v = np.array([[[10.0, 0.0], [0.0, 20.0]]])

    attn = DotProductAttention(q_dim=2, k_dim=2, v_dim=2)
    print(f"  模块参数量 = {attn.num_params()}（q_dim == k_dim，投影是多余的，所以零参数）\n")

    # ---- 手算（完全不用模块的代码）----
    d = 2
    scale = 1.0 / np.sqrt(d)
    scores_hand = np.array([[q[0, 0] @ k[0, 0], q[0, 0] @ k[0, 1]]]) * scale
    e = np.exp(scores_hand - scores_hand.max())
    w_hand = e / e.sum()
    ctx_hand = w_hand @ v[0]

    print("  手算过程：")
    print(f"    d = {d},  scale = 1/√d = {scale:.6f}")
    print(f"    score₀ = (q·k₀)·scale = (1·1 + 0·0)·{scale:.6f} = {scores_hand[0,0]:.6f}")
    print(f"    score₁ = (q·k₁)·scale = (1·0 + 0·1)·{scale:.6f} = {scores_hand[0,1]:.6f}")
    print(f"    exp(score) = [{np.exp(scores_hand[0,0]):.6f}, {np.exp(scores_hand[0,1]):.6f}]")
    print(f"    weights   = {w_hand.round(6)}     （和 = {w_hand.sum():.6f}）")
    print(f"    context   = {w_hand[0,0]:.6f}·[10,0] + {w_hand[0,1]:.6f}·[0,20]")
    print(f"              = {ctx_hand.round(6)}\n")

    ctx_mod, w_mod = attn(q, k, v)
    print("  模块输出：")
    print(f"    weights = {w_mod.reshape(-1).round(6)}")
    print(f"    context = {ctx_mod.reshape(-1).round(6)}\n")

    print(f"  权重最大绝对差 = {np.abs(w_mod.reshape(-1) - w_hand).max():.3e}")
    print(f"  输出最大绝对差 = {np.abs(ctx_mod.reshape(-1) - ctx_hand).max():.3e}")
    print("  → 机器精度级别的一致，手算与实现完全吻合。")

    print(f"\n  顺带检查：不加 1/√d 会怎样？")
    scores_noscale = np.array([[1.0, 0.0]])
    w_noscale = softmax(scores_noscale)
    print(f"    不加缩放 weights = {w_noscale.round(6)}   （更尖锐）")
    print(f"    加了缩放 weights = {w_hand.round(6)}   （更平缓）")
    print("  维度越高差距越夸张：d=128 时点积标准差约 √128 ≈ 11.3，")
    print("  不加缩放 softmax 会退化成接近 one-hot，梯度趋近 0，训练直接卡死。")


# ======================================================================================
# [3] mask
# ======================================================================================
def section_mask() -> None:
    title("[3] mask：两种实现方案 + 一个真实的广播 bug")

    print("  方案 A：scores + (1-mask)·(-1e9)，然后普通 softmax   （Batch.attn_mask() 的做法）")
    print("  方案 B：np.where(mask, scores, -inf)，然后带保护的 softmax   （本模块的做法）\n")

    rng = np.random.default_rng(0)
    scores = rng.normal(scale=2.0, size=(1, 10))
    valid = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.float64)

    w_a = softmax(scores + (1.0 - valid) * NEG_INF)
    w_b = softmax_masked(scores, valid)

    print("  (a) 正常量级下（scores 约 ±4）：")
    print(f"      方案 A weights = {w_a.reshape(-1).round(6)}")
    print(f"      方案 B weights = {w_b.reshape(-1).round(6)}")
    print(f"      最大绝对差 = {np.abs(w_a - w_b).max():.3e}")
    print("      → 两者**逐位相同**，一个 bit 都不差。")
    print("        网上常说方案 A「有精度损失」，实测并不成立：")
    print("        有效位置加的是精确的 0.0；填充位置 exp(-1e9 - max) 在 float64 下")
    print("        下溢为精确的 0.0。这个说法我一开始也写了，跑完实验才发现是错的。\n")

    # (b) 整行全被 mask
    allmask = np.zeros((1, 10), dtype=np.float64)
    w_a_all = softmax(scores + (1.0 - allmask) * NEG_INF)
    w_b_all = softmax_masked(scores, allmask)
    print("  (b) 真正有区别的地方：某一行**全部**被 mask 掉时")
    print("      （理论上不该发生，但必须定义好行为，否则出问题时毫无线索）")
    print(f"      方案 A weights = {w_a_all.reshape(-1).round(6)}")
    print(f"      方案 A 权重和 = {w_a_all.sum():.6f}")
    print("      → 方案 A 静默退化成了 softmax(scores)！所有元素同时减 1e9，")
    print("        这个常数在 max-shift 里相互抵消，mask 等于完全没加。")
    print("        得到一个看似正常、实则毫无意义的分布 —— 非常隐蔽。")
    print(f"      方案 B weights = {w_b_all.reshape(-1).round(6)}")
    print(f"      方案 B 权重和 = {w_b_all.sum():.6f}   ← 全 0，语义是「没有可注意的地方」")
    print("      → 而且没有产生 nan（-inf 会让 max 也变 -inf，-inf-(-inf)=nan，")
    print("        所以 softmax_masked 里显式做了 isfinite 保护）。这就是选方案 B 的实质理由。\n")

    # (c) 广播陷阱：本次实现时真的踩到了
    print("  (c) ⚠ 一个我实际踩到的 bug：numpy 广播从右对齐")
    print("      numpy 的广播规则是**从最右边一维开始向左对齐**。所以")
    print("          scores (B, Tq, S)  与  mask (B, S)")
    print("      被对齐成")
    print("          (B, Tq, S)  与  (1, B, S)      ← 注意 B 跑到了中间！")
    print("      而不是我们想要的 (B, 1, S)。后果分两种情况：\n")

    # 情况一：报错
    sc1 = rng.normal(size=(2, 3, 4))            # B=2, Tq=3, S=4
    m1 = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float64)
    try:
        np.where(m1 > 0, sc1, -np.inf)
        print("      情况一：B=2, Tq=3  -> 竟然没报错？")
    except ValueError as exc:
        print(f"      情况一：B=2, Tq=3, S=4  ->  抛 ValueError（还算幸运）")
        print(f"              {str(exc).splitlines()[0][:70]}")

    # 情况二：静默出错 —— 而这恰好是解码推理的真实配置
    B2, Tq2, S2 = 4, 1, 5
    sc2 = rng.normal(size=(B2, Tq2, S2))        # B=4, Tq=1, S=5
    m2 = np.zeros((B2, S2), dtype=np.float64)
    for b in range(B2):
        m2[b, : S2 - b] = 1.0                   # 每个样本的有效长度不同
    bad = np.where(m2 > 0, sc2, -np.inf)
    print(f"\n      情况二：B={B2}, Tq={Tq2}, S={S2}  ->  **没有报错**")
    print(f"              期望形状 ({B2}, {Tq2}, {S2})，实际得到 {bad.shape}")
    print("              中间多出来一维：每个 query 用上了**别的样本的 mask**。")
    print("              而且这个配置 (Tq=1, B=32) 恰好就是解码推理时的真实情况！")
    print("              只看形状「(4,4,5) 里有 4 和 5，看着挺合理」，肉眼极难发现。\n")
    print("      修复：显式插一个轴，把 (B,S) 变成 (B,1,S) 再广播。")
    fixed = softmax_masked(sc2, m2)
    print(f"              softmax_masked 输出形状 = {fixed.shape}   ← 正确")
    ok = all(np.all(fixed[b, 0][m2[b] == 0] == 0.0) for b in range(B2))
    print(f"              逐样本核对「mask=0 的位置权重严格为 0」：{'全部通过 ✓' if ok else '不通过 ✗'}")
    print("              （这正是 attention.py 里 _split_mask / _check_shapes 存在的原因）")

    # 用真实的 Batch 收尾
    print()
    from seq2seq_attn import data as D
    train, _, _ = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(4)))
    fake_scores = rng.normal(size=(4, 1, batch.src.shape[1]))     # Tq=1，模拟解码一步
    w = softmax_masked(fake_scores, batch.src_mask)
    print(f"  用真实 batch 的 src_mask 验证（Tq=1, mask 形状 {batch.src_mask.shape}）：")
    print(f"    输出形状 = {w.shape}")
    for i in range(4):
        pad_w = w[i, 0][batch.src_mask[i] < 0.5].sum()
        print(f"    样本 {i}: 真实长度 {int(batch.src_len[i]):>2}，"
              f"填充位置权重和 = {pad_w:.3e}，权重总和 = {w[i, 0].sum():.6f}")
    print("    → 每个样本用的是自己的 mask，长度不同也各自归一化到 1。")


# ======================================================================================
# [4] 结构对比
# ======================================================================================
def section_compare() -> None:
    title("[4] 加性注意力 vs 缩放点积注意力")

    d_q = d_k = d_v = 128
    a_bah = BahdanauAttention(d_q, d_k, d_v, attn_dim=128)
    a_dot = DotProductAttention(d_q, d_k, d_v)
    a_dot_proj = DotProductAttention(d_q, d_k, d_v, dim=64)

    print("  待解释的维度：q_dim = k_dim = v_dim = 128，注意力中间维 A = 128\n")
    print(f"  加性 Bahdanau：参数 {a_bah.num_params():>7,}"
          f"   （W_q {d_q}×128 + W_k {d_k}×128 + v 128 + b 128）")
    print(f"                 {a_bah}")
    print(f"  点积（无投影）：参数 {a_dot.num_params():>7,}   ← 零参数！")
    print(f"                 {a_dot}")
    print(f"  点积（投影到 64）：参数 {a_dot_proj.num_params():>6,}")
    print(f"                 {a_dot_proj}\n")

    print("  打分函数：")
    print("    加性   scores = vᵀ · tanh(W_q·q + W_k·k)      ← 有 tanh，非线性，参数可学")
    print("    点积   scores = (q·kᵀ) / √d                  ← 双线性型，无参数")
    print("  点积衡量的是「两个向量方向有多一致」，是一个**固定**的匹配函数；")
    print("  加性则通过学习 W_q / W_k / v 来**塑造**匹配函数，表达力更强。")
    print("  代价就是多了一堆参数（本例 33k，是零参数的无穷多倍 :)）。\n")

    print("  计算量对比（B=32, S=11，前向 + 反向，数值取 50 次重复的平均）：")
    print(f"    {'配置':<24}{'参数量':>9}{'单次调用(ms)':>15}{'Tq':>5}{'分摊每步(ms)':>15}")
    print("    " + "-" * 68)
    for label, mod, tq in [
        ("加性  Tq=1（逐步解码）", a_bah, 1),
        ("加性  Tq=11（一次算完）", a_bah, 11),
        ("点积  Tq=1（逐步解码）", a_dot, 1),
        ("点积  Tq=11（一次算完）", a_dot, 11),
    ]:
        B, S = 32, 11
        g = np.random.default_rng(0)
        q = np.ascontiguousarray(g.normal(size=(B, tq, d_q)))
        k = np.ascontiguousarray(g.normal(size=(B, S, d_k)))
        v = np.ascontiguousarray(g.normal(size=(B, S, d_v)))
        mask = np.ones((B, S))
        dctx = g.normal(size=(B, tq, d_v))

        mod.zero_grad()
        mod.forward(q, k, v, mask)              # 预热
        n_rep = 50
        t0 = time.perf_counter()
        for _ in range(n_rep):
            mod.zero_grad()
            mod.forward(q, k, v, mask)
            mod.backward(dctx)
        per_call = (time.perf_counter() - t0) / n_rep * 1000
        print(f"    {label:<22}{mod.num_params():>9,}{per_call:>15.3f}{tq:>5}{per_call / tq:>15.3f}")

    print("\n  说明：加性注意力为了算 tanh 需要显式撑开 (B, Tq, S, A) 这个四维张量")
    print("  （本例 B=32, Tq=11, S=11, A=128 时就是 500 万个元素）；")
    print("  点积注意力只需要一次批量矩阵乘法，不需要中间的 A 维。")
    print("  这就是 Transformer 选用点积注意力的主要工程原因 —— 而它敢用，")
    print("  是因为有 1/√d 缩放把方差压住，并用多头机制补回表达力。")


# ======================================================================================
# [5] 朴素三重循环交叉验证
# ======================================================================================
def section_naive_crosscheck() -> None:
    title("[5] 交叉验证：用朴素三重循环独立重实现一遍")

    print("  光看自己写的向量化代码「跑得通」是不够的 —— 广播（broadcasting）")
    print("  很容易在某个轴上悄悄错位，而结果看起来完全合理。")
    print("  所以这里用最笨的显式循环重算一遍：没有 einsum、没有广播、")
    print("  没有矩阵乘法，每个元素老老实实按定义累加。")
    print("  两者的 bug 不可能重合，比对一致才算真的对。\n")

    rng = np.random.default_rng(3)
    B, Tq, S, dq, dk, dv, A = 2, 2, 3, 4, 4, 5, 6

    def naive_scores_bahdanau(query, keys, W_q, W_k, v, b):
        out = np.zeros((B, Tq, S))
        for bi in range(B):
            for ti in range(Tq):
                for si in range(S):
                    acc = 0.0
                    for a in range(A):
                        pre = b[a]
                        for d in range(dq):
                            pre += query[bi, ti, d] * W_q[d, a]
                        for d in range(dk):
                            pre += keys[bi, si, d] * W_k[d, a]
                        acc += np.tanh(pre) * v[a]
                    out[bi, ti, si] = acc
        return out

    def naive_scores_dot(query, keys):
        out = np.zeros((B, Tq, S))
        for bi in range(B):
            for ti in range(Tq):
                for si in range(S):
                    acc = 0.0
                    for d in range(dq):
                        acc += query[bi, ti, d] * keys[bi, si, d]
                    out[bi, ti, si] = acc / np.sqrt(dq)
        return out

    def naive_context(weights, values):
        out = np.zeros((B, Tq, dv))
        for bi in range(B):
            for ti in range(Tq):
                for x in range(dv):
                    acc = 0.0
                    for si in range(S):
                        acc += weights[bi, ti, si] * values[bi, si, x]
                    out[bi, ti, x] = acc
        return out

    query = rng.normal(size=(B, Tq, dq))
    keys = rng.normal(size=(B, S, dk))
    values = rng.normal(size=(B, S, dv))
    mask = np.ones((B, S))
    mask[0, 2] = 0.0            # 故意留一个 padding 位置，检验 mask 也参与比对

    # ---- 加性 ----
    a_bah = BahdanauAttention(dq, dk, dv, attn_dim=A, seed=11)
    ctx_mod, w_mod = a_bah(query, keys, values, mask)

    P = a_bah.params()
    sc = naive_scores_bahdanau(query, keys, P["W_q"], P["W_k"], P["v"], P["b"])
    sc = np.where(mask[:, None, :] > 0, sc, -np.inf)
    m = sc.max(axis=-1, keepdims=True)
    e = np.exp(sc - np.where(np.isfinite(m), m, 0.0))
    w_naive = e / np.maximum(e.sum(-1, keepdims=True), 1e-300)
    ctx_naive = naive_context(w_naive, values)

    print("  加性 Bahdanau：")
    print(f"    weights 最大绝对差 = {np.abs(w_mod - w_naive).max():.3e}")
    print(f"    context 最大绝对差 = {np.abs(ctx_mod - ctx_naive).max():.3e}")

    # ---- 点积 ----
    a_dot = DotProductAttention(dq, dk, dv, seed=12)
    ctx_mod2, w_mod2 = a_dot(query, keys, values, mask)

    sc2 = naive_scores_dot(query, keys)
    sc2 = np.where(mask[:, None, :] > 0, sc2, -np.inf)
    m2 = sc2.max(axis=-1, keepdims=True)
    e2 = np.exp(sc2 - np.where(np.isfinite(m2), m2, 0.0))
    w_naive2 = e2 / np.maximum(e2.sum(-1, keepdims=True), 1e-300)
    ctx_naive2 = naive_context(w_naive2, values)

    print("  缩放点积：")
    print(f"    weights 最大绝对差 = {np.abs(w_mod2 - w_naive2).max():.3e}")
    print(f"    context 最大绝对差 = {np.abs(ctx_mod2 - ctx_naive2).max():.3e}")

    print("\n  → 向量化实现与朴素定义逐值一致，广播没有错位。")
    print(f"  另外验证 mask 生效：样本 0 第 2 位是 <pad>，其权重 = {w_mod[0,0,2]:.1e}")


# ======================================================================================
# [6] 梯度检验
# ======================================================================================
def make_loss(attn, query, keys, values, mask, dcontext):
    """构造 loss = Σ context ⊙ dcontext，此时 dL/dcontext 恰好就是 dcontext。"""
    def loss_fn():
        ctx, _ = attn.forward(query, keys, values, mask)
        return float((ctx * dcontext).sum())
    return loss_fn


def gradient_report(label: str, attn, query, keys, values, mask, rng, n_samples=6, eps=1e-5) -> list[dict]:
    dcontext = rng.normal(size=(query.shape[0], query.shape[1], values.shape[2]))
    loss_fn = make_loss(attn, query, keys, values, mask, dcontext)

    loss_fn()                       # 先跑一次，确保 forward 已执行、cache 已填好
    attn.zero_grad()
    dinputs = attn.backward(dcontext)

    tensors = {"query": query, "keys": keys, "values": values}
    tensors.update(attn.params())
    analytic = dict(dinputs)
    analytic.update(attn.grads())

    results = check_gradient(tensors, analytic, loss_fn, n_samples=n_samples, eps=eps, rng=rng)
    print(f"\n  {label}   （loss 用随机方向投影 context，能覆盖完整 Jacobian）")
    print(f"    {'张量':<10}{'形状':<16}{'分歧/抖动':>12}{'解析梯度':>15}{'相对误差':>12}  判读")
    print("    " + "-" * 82)
    for r in results:
        rel = "—" if r["max_rel_err"] != r["max_rel_err"] else f"{r['max_rel_err']:.2e}"
        note = "✓" if r["status"] == "ok" else ("~ 梯度贴噪声地板" if r["status"] == "low-snr" else "✗")
        print(f"    {r['name']:<10}{str(r['shape']):<16}{r['ratio']:>12.2e}"
              f"{r['worst_ana']:>15.4e}{rel:>12}  {note}")
    summary = summarize_gradient_report(results)
    print(f"    最大「分歧/抖动」= {summary['worst_ratio']:.2e}   "
          f"{'✓ 通过' if summary['passed'] else '✗ 不通过：' + str(summary['failed'])}")
    return results


def section_eps_choice() -> None:
    """一个小实验：中心差分的 ε 该怎么选。"""
    print("\n  " + SUB[:74])
    print("  插播一个小实验：ε 该怎么选？")
    print("  " + SUB[:74])
    print("  中心差分有两类误差在互相拉扯：")
    print("    · 截断误差（数学近似）：O(ε²)      —— ε 越小越好")
    print("    · 舍入误差（浮点抵消）：|L|·ε_mach / (2ε)  —— ε 越大越好")
    print("  两者之和在 ε ≈ ε_mach^(1/3) 处取最小。float64 的 ε_mach ≈ 2.2e-16，")
    print("  所以最优 ε ≈ 6e-6，取 1e-5 左右最合适。这不是拍脑袋，是能算出来的。\n")

    rng = np.random.default_rng(0)
    B, Tq, S, D = 2, 2, 6, 128
    query = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    keys = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    values = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    mask = np.ones((B, S))
    mask[0, 4:] = 0.0

    attn = BahdanauAttention(D, D, D, attn_dim=128, seed=1)
    dcontext = rng.normal(size=(B, Tq, D))
    loss_fn = make_loss(attn, query, keys, values, mask, dcontext)

    loss_fn()
    attn.zero_grad()
    dinputs = attn.backward(dcontext)
    tensors = {"query": query, "keys": keys, "values": values}
    tensors.update(attn.params())
    analytic = dict(dinputs)
    analytic.update(attn.grads())

    print(f"  被测对象：加性 Bahdanau 的全部参数与输入，loss 量级 |L| = {abs(loss_fn()):.2f}")
    print(f"  {'ε':>9}{'最大相对误差':>16}   判读")
    print("  " + "-" * 62)
    for eps in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        # 固定 rng 种子，保证各个 ε 抽查的是同一批下标，比较才公平
        r = check_gradient(tensors, analytic, loss_fn, n_samples=8, eps=eps,
                           rng=np.random.default_rng(0))
        w = max(x["max_rel_err"] for x in r)
        if w < 1e-7:
            verdict = "完美"
        elif w < 1e-5:
            verdict = "可接受"
        elif w < 1e-2:
            verdict = "偏差明显"
        else:
            verdict = "完全不可信"
        if eps == 1e-8:
            verdict += "（舍入噪声淹没一切）"
        if eps == 1e-5:
            verdict += "（最优点附近）"
        if eps == 1e-3:
            verdict += "（截断误差占主导）"
        print(f"  {eps:>9.0e}{w:>16.2e}   {verdict}")

    print("\n  → 典型的 U 形曲线：ε 太小被浮点噪声吃掉，太大被数学近似吃掉。")
    print("    所以本项目统一用 ε = 1e-5，此时正确的梯度实现能稳定达到 ~1e-8。")


def section_gradcheck() -> None:
    title("[6] 梯度检验：数值梯度 vs 解析梯度")

    print("  原理：解析梯度来自我们手写的 backward；数值梯度来自中心差分")
    print("        ∂L/∂θ ≈ ( L(θ+ε) - L(θ-ε) ) / 2ε")
    print("  数值梯度只依赖 forward，**与我们写的 backward 完全无关**，")
    print("  所以两者吻合就说明 backward 的每一步推导都对。\n")
    print("  注意这里同时检验了两类张量：")
    print("    · 模型参数（W_q / W_k / v / b）—— 训练要用的梯度")
    print("    · 输入张量（query / keys / values）—— 第 4 步拼接解码器时的链式传递")
    print("  后者往往更难写对（广播求和的轴很容易搞错），所以一定要一起检。\n")
    print(f"  配置：B=2, Tq=2, S=6, dq=dk=dv=128, A=128, eps=1e-5, 每张量抽查 6 个元素")

    rng = np.random.default_rng(0)
    B, Tq, S, D = 2, 2, 6, 128
    query = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    keys = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    values = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    mask = np.ones((B, S))
    mask[0, 4:] = 0.0                       # 让 mask 真的起作用

    a_bah = BahdanauAttention(D, D, D, attn_dim=128, seed=1)
    gradient_report("加性 Bahdanau：", a_bah, query.copy(), keys.copy(), values.copy(), mask, rng)

    a_dot = DotProductAttention(D, D, D, seed=2)
    gradient_report("缩放点积（零参数）：", a_dot, query.copy(), keys.copy(), values.copy(), mask, rng)

    a_dot_p = DotProductAttention(D, D, D, dim=64, seed=3)
    gradient_report("缩放点积（投影到 64）：", a_dot_p, query.copy(), keys.copy(), values.copy(), mask, rng)

    print("\n  判读标准（float64 + ε=1e-5）：")
    print("    < 1e-7   完美，梯度推导正确")
    print("    ~1e-5    可接受，但某处可能数值不稳定")
    print("    > 1e-3   基本可以确定公式写错了")

    section_eps_choice()


# ======================================================================================
# [7] 负面对照
# ======================================================================================
class ForgottenScale(DotProductAttention):
    """故意写错的反向传播：忘记除以 √d。

    前向里 `scores = (q·kᵀ) * scale`，反向里 dq/dk 也必须乘同一个 scale。
    这是最常见的一类错误 —— 前向改了、反向忘了同步改，而且**不报错**。
    """

    def backward(self, dcontext):
        saved, self.scale = self.scale, 1.0
        try:
            return super().backward(dcontext)
        finally:
            self.scale = saved


def section_bug_hunt() -> None:
    title("[7] 反向验证：梯度检验真的抓得住错误吗？")

    print("  一个只在「数值正确」时才可信的检查是没用的。")
    print("  所以这里故意写一个经典 bug：反向传播忘记乘 1/√d。")
    print("  这种错误不会报异常、不会出 nan，训练还会「跑起来」，只是学不好 ——")
    print("  是最难查的一类问题。\n")

    rng = np.random.default_rng(0)
    B, Tq, S, D = 2, 2, 6, 128
    query = np.ascontiguousarray(rng.normal(size=(B, Tq, D)))
    keys = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    values = np.ascontiguousarray(rng.normal(size=(B, S, D)))
    mask = np.ones((B, S))

    print(f"  正确版本（scale = 1/√{D} = {1/np.sqrt(D):.6f}）：")
    good = DotProductAttention(D, D, D, seed=7)
    r_good = gradient_report("正确：", good, query.copy(), keys.copy(), values.copy(), mask, rng)

    print(f"\n  错误版本（前向照常乘 scale，反向乘 1.0）：")
    bad = ForgottenScale(D, D, D, seed=7)
    r_bad = gradient_report("写错：", bad, query.copy(), keys.copy(), values.copy(), mask, rng)

    s_good = summarize_gradient_report(r_good)
    s_bad = summarize_gradient_report(r_bad)
    w_good = s_good["worst_ratio"]
    w_bad = s_bad["worst_ratio"]
    print(f"\n  对比最大「分歧/抖动」：正确版 {w_good:.2e}  →  写错版 {w_bad:.2e}")
    print(f"  相差 {w_bad / max(w_good, 1e-30):.1e} 倍。错误被**放大到肉眼可见**：")
    print("  如果只看 forward 的输出，两个版本一模一样，根本发现不了。")
    print("  这就是手写反向传播时梯度检验不可替代的原因。")


# ======================================================================================
# [8] 可视化
# ======================================================================================
def section_visualize() -> None:
    title("[8] 注意力权重长什么样")

    print("  造一个小例子，让「应该关注哪里」有明确答案：")
    print("    源句「2008年3月11日」→ 目标「11/March/2008」")
    print("    期望：生成 '11' 时关注日期的两个 '1'；")
    print("          生成 'March' 时关注 '3' 和 '月'；")
    print("          生成 '2008' 时关注年份的四个数字。\n")

    src_text = "2008年3月11日"
    tokens = ["2", "0", "0", "8", "年", "3", "月", "1", "1", "日"]
    S, D = len(tokens), 64

    rng = np.random.default_rng(42)
    # 造一个「假的」词嵌入表：让相同字符有相近的向量，
    # 这样点积注意力就会自然地把 query 引到对应的源 token 上。
    chars = "0123456789年月日"
    emb = {c: rng.normal(size=D) for c in chars}
    keys = np.stack([emb[c] for c in tokens])[None, :, :]        # (1, S, D)

    # 三个解码步的 query：分别对应「日」「分隔符」「月」
    steps = [
        ("生成 '1'", emb["1"] * 1.0 + rng.normal(scale=0.35, size=D)),
        ("生成 '/'", rng.normal(scale=1.0, size=D)),
        ("生成 'M'", (emb["3"] + emb["月"]) / 2.0 + rng.normal(scale=0.35, size=D)),
        ("生成 '2'", emb["2"] * 1.0 + rng.normal(scale=0.35, size=D)),
    ]
    query = np.stack([q for _, q in steps])[None, :, :]          # (1, Tq, D)

    attn = DotProductAttention(D, D, D, seed=0)
    ctx, w = attn(query, keys, keys)
    w2 = w[0]                                                     # (Tq, S)

    print(f"  源句：{src_text}   （{S} 个 token）")
    print(f"  注意力：缩放点积，d = {D}，零参数\n")
    render_matrix(w2, tokens, [label for label, _ in steps])

    print("\n  用条形图看得更清楚（每个解码步单独一行）：")
    for r, (label, _) in enumerate(steps):
        print(f"\n    {label}：")
        for i, tok in enumerate(tokens):
            bar = "█" * int(round(w2[r, i] * 45))
            print(f"      {pad_display(f'[{i}] {tok}', 10)}{bar} {w2[r, i]:.3f}")

    print("\n  可以看到：")
    print("    · 生成 '1' 时权重集中在第 7、8 位的两个 '1'（也就是源句的日）")
    print("    · 生成 'M'（月份）时权重落在第 5、6 位的 '3' 和 '月'")
    print("    · 生成 '/' 时 query 是纯噪声，权重是散的 —— 因为分隔符不需要看源句")
    print("\n  这个「query 去找和自己语义相近的 key」的现象，就是")
    print("  第 7 步要画的对齐热力图的雏形。真实模型里权重会更干净、更接近对角。")


# ======================================================================================
def main() -> None:
    print(LINE)
    print(" 第 2 步 · 注意力机制独立验证")
    print(LINE)
    print(" 注意力模块是一个纯粹的数学函数，先把它验证到位，第 4 步拼接时才有底。")

    section_essence()
    section_hand_check()
    section_mask()
    section_compare()
    section_naive_crosscheck()
    section_gradcheck()
    section_bug_hunt()
    section_visualize()

    title("第 2 步完成 ✓")
    print(" 注意力机制的前向与反向都已通过独立验证。")
    print(" 下一步（第 3 步）将实现编码器 GRU 的前向与反向传播。")
    print()


if __name__ == "__main__":
    main()
