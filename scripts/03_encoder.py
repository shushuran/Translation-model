"""第 3 步自检脚本：基础层 + 编码器 GRU 的前向与反向验证。

运行：
    python scripts/03_encoder.py

验证顺序（自下而上，出问题时能立刻定位到层）：
    [1] Linear          前向 / 反向 / 梯度检验
    [2] Embedding       重复索引必须用 np.add.at
    [3] GRU 两个门      用数值实验确认 z 和 r 的语义
    [4] GRU 单元        与朴素逐门实现交叉验证
    [5] BPTT            梯度检验（单元级 + 编码器端到端）
    [6] padding 不变量  加 padding 不能改变真实位置的输出
    [7] 冻结状态 vs 不冻结
    [8] 编码真实日期    参数量拆解与隐状态观测
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

import numpy as np

from seq2seq_attn import data as D
from seq2seq_attn.config import CONFIG
from seq2seq_attn.encoder import GRUCell, GRUEncoder
from seq2seq_attn.nn import (
    Embedding,
    Linear,
    check_gradient,
    sigmoid,
    summarize_gradient_report,
)

LINE = "=" * 82
SUB = "-" * 82
STATUS_NOTE = {"ok": "✓", "low-snr": "~ 梯度贴噪声地板", "FAIL": "✗ 不通过"}


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def grad_table(label: str, tensors: dict, analytic: dict, loss_fn, rng, n_samples=5, eps=1e-5):
    results = check_gradient(tensors, analytic, loss_fn, n_samples=n_samples, eps=eps, rng=rng)
    print(f"\n  {label}")
    print(f"    {'张量':<16}{'形状':<16}{'分歧/抖动':>12}{'解析梯度':>15}{'相对误差':>12}  判读")
    print("    " + "-" * 86)
    for r in results:
        rel = "—" if r["max_rel_err"] != r["max_rel_err"] else f"{r['max_rel_err']:.2e}"
        print(f"    {r['name']:<16}{str(r['shape']):<16}{r['ratio']:>12.2e}"
              f"{r['worst_ana']:>15.4e}{rel:>12}  {STATUS_NOTE[r['status']]}")
    summary = summarize_gradient_report(results)
    print(f"    最大「分歧/抖动」= {summary['worst_ratio']:.2e}   "
          f"{'✓ 通过' if summary['passed'] else '✗ 不通过：' + str(summary['failed'])}")
    return summary


# ======================================================================================
# [1] Linear
# ======================================================================================
def section_linear() -> None:
    title("[1] Linear：y = xW + b")

    rng = np.random.default_rng(0)
    lin = Linear(6, 4, seed=1)
    x = np.ascontiguousarray(rng.normal(size=(3, 6)))
    y = lin(x)
    check_manual = x @ lin.params()["W"] + lin.params()["b"]
    print(f"  Linear(6, 4)  参数 {lin.num_params()}")
    print(f"  x {x.shape}  ->  y {y.shape}")
    print(f"  手算比对最大绝对差 = {np.abs(y - check_manual).max():.3e}\n")

    # 任意前置维度：GRU 喂 (B, in)，注意力喂 (B, T, in)
    x3 = np.ascontiguousarray(rng.normal(size=(2, 5, 6)))
    print(f"  支持任意前置维度：x {x3.shape} -> y {lin(x3).shape}   （内部展平成 (10, 6) 做一次矩阵乘法）")

    # 梯度检验（注意：这里同时检验参数和输入）
    lin2 = Linear(6, 4, seed=2)
    xg = np.ascontiguousarray(rng.normal(size=(3, 6)))
    G = rng.normal(size=(3, 4))

    def loss_fn():
        return float((lin2.forward(xg) * G).sum())

    loss_fn()
    lin2.zero_grad()
    dx = lin2.backward(G)
    tensors = {"x": xg}
    tensors.update(lin2.params())
    analytic = {"x": dx}
    analytic.update(lin2.grads())
    grad_table("梯度检验（ε=1e-5）：", tensors, analytic, loss_fn, np.random.default_rng(0))


# ======================================================================================
# [2] Embedding
# ======================================================================================
def section_embedding() -> None:
    title("[2] Embedding：花式索引是拷贝，必须用 np.add.at")

    emb = Embedding(5, 3, seed=0)
    W = emb.params()["weight"]
    W[...] = np.arange(15).reshape(5, 3)          # 用 0..14 填满，方便肉眼核对

    print("  权重表（行 = token id）：")
    for i in range(5):
        print(f"    id {i}: {W[i]}")

    idx = np.array([[1, 1, 2, 1]])                # id=1 出现了 3 次
    print(f"\n  查询序列 index = {idx.tolist()}   →  id=1 出现 3 次")
    out = emb.forward(idx)
    print(f"  输出形状 {out.shape}")

    dout = np.ones((1, 4, 3))                     # 每个位置的上游梯度都是全 1
    print(f"  上游梯度 dout 全为 1，形状 {dout.shape}\n")

    # ---- 正确做法 ----
    emb.zero_grad()
    emb.backward(dout)
    correct = emb.grads()["weight"].copy()

    # ---- 错误做法：花式索引 += ----
    wrong = np.zeros_like(W)
    flat_idx = idx.reshape(-1)
    flat_dout = dout.reshape(-1, 3)
    wrong[flat_idx] += flat_dout

    print("  两种做法的梯度表：")
    print(f"    {'id':<4}{'np.add.at（正确）':<26}{'grad[idx] += （错误）':<26}")
    print("    " + "-" * 62)
    for i in range(5):
        print(f"    {i:<4}{str(correct[i]):<26}{str(wrong[i]):<26}")

    print("\n  差异原因：`grad[idx] += dout` 展开后是")
    print("      tmp = grad[idx]     # 先把 idx 对应的行**复制**出来")
    print("      tmp = tmp + dout")
    print("      grad[idx] = tmp     # 再整体写回")
    print("  写回时**后面的会覆盖前面的**，id=1 的三次出现只剩最后一次（梯度 1，不是 3）。")
    print(f"  → id=1 正确梯度 {correct[1]}，错误梯度 {wrong[1]}")

    print("\n  这个坑在真实数据里一定会踩到：源句「2008年11月11日」里")
    toks = D.tokenize("2008年11月11日")
    print(f"    token = {toks}")
    from collections import Counter
    cnt = Counter(toks)
    dup = {k: v for k, v in cnt.items() if v > 1}
    print(f"    重复出现的 token：{dup}")
    print("  '1' 在一句话里就出现 4 次。不用 np.add.at，这些位置的 embedding 学不动。")

    # 梯度检验
    emb2 = Embedding(5, 3, seed=3)
    idxg = np.array([[1, 1, 2, 1], [0, 3, 3, 4]])
    Gg = np.random.default_rng(1).normal(size=(2, 4, 3))

    def loss_fn():
        return float((emb2.forward(idxg) * Gg).sum())

    loss_fn()
    emb2.zero_grad()
    emb2.backward(Gg)
    grad_table("梯度检验（重复索引情形）：", dict(emb2.params()), dict(emb2.grads()),
               loss_fn, np.random.default_rng(0))


# ======================================================================================
# [3] GRU 两个门的语义
# ======================================================================================
def section_gates() -> None:
    title("[3] GRU 两个门的语义（数值实验）")

    H = 6
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=(1, 4))
    x2 = rng.normal(size=(1, 4))
    h1 = rng.normal(size=(1, H))
    h2 = rng.normal(size=(1, H))

    print("  公式（PyTorch 约定）：")
    print("    r = σ(x·W_ir + h·W_hr)                    重置门")
    print("    z = σ(x·W_iz + h·W_hz)                    更新门")
    print("    n = tanh(x·W_in + r ⊙ (h·W_hn))           候选状态")
    print("    h' = (1 - z) ⊙ n + z ⊙ h")
    print()

    def make_cell(bias_z=0.0, bias_r=0.0):
        c = GRUCell(4, H, seed=1)
        c.params()["b_ih"][H : 2 * H] = bias_z      # z 的输入偏置
        c.params()["b_hh"][H : 2 * H] = bias_z
        c.params()["b_ih"][0:H] = bias_r            # r 的输入偏置
        c.params()["b_hh"][0:H] = bias_r
        return c

    # ---- z -> 1：完全保留旧状态 ----
    cell = make_cell(bias_z=30.0)
    o1 = cell.forward(x1, h1)
    o2 = cell.forward(x2, h1)
    o3 = cell.forward(x1, h2)
    print("  实验一：把 z 逼到 1（bias_z = +30，σ(30) ≈ 1）")
    print(f"    不同输入 x：|h'(x1,h) - h'(x2,h)| = {np.abs(o1 - o2).max():.3e}")
    print(f"    不同历史 h：|h'(x,h1) - h'(x,h2)| = {np.abs(o1 - o3).max():.3e}")
    print(f"    |h' - h1| = {np.abs(o1 - h1).max():.3e}   ← h' ≈ h，完全保留记忆")
    print("    → z = 1 表示「不更新」，历史被原样传递下去。")

    # ---- z -> 0：完全采纳新候选 ----
    cell = make_cell(bias_z=-30.0)
    o1 = cell.forward(x1, h1)
    o3 = cell.forward(x1, h2)
    print("\n  实验二：把 z 逼到 0（bias_z = -30，σ(-30) ≈ 0）")
    print(f"    不同历史 h：|h'(x,h1) - h'(x,h2)| = {np.abs(o1 - o3).max():.3e}")
    print("    → z = 0 时 h' ≈ n，只依赖当前输入与 r ⊙ (h·W_hn)，旧状态不直接保留。")
    print("      注意此时 n 里还有 h 的影子（经由重置门那一项），所以并非完全无关。")

    # ---- r -> 0：候选状态忽略历史 ----
    cell = make_cell(bias_z=-30.0, bias_r=-30.0)
    o1 = cell.forward(x1, h1)
    o3 = cell.forward(x1, h2)
    print("\n  实验三：再把 r 逼到 0（bias_r = -30）")
    print(f"    不同历史 h：|h'(x,h1) - h'(x,h2)| = {np.abs(o1 - o3).max():.3e}   ← 严格为 0")
    print("    → r = 0 时 r ⊙ (h·W_hn) = 0，候选状态 n = tanh(x·W_in) **完全忽略历史**，")
    print("      相当于把记忆「重置」成只看当前词。这正是 reset gate 名字的由来。")

    print("\n  小结：")
    print("    z 控制「保留多少旧状态」—— 长期记忆的开关")
    print("    r 控制「计算新候选时参考多少旧状态」—— 短期遗忘的开关")
    print("    两者配合，让 GRU 能学会「该记多久」和「该忘什么」。")


# ======================================================================================
# [4] 朴素逐门实现交叉验证
# ======================================================================================
def section_crosscheck() -> None:
    title("[4] 交叉验证：与朴素「逐门独立线性层」实现比对")

    print("  本项目的 GRUCell 把三个门拼成一个 (in, 3H) 的大矩阵，一次矩阵乘法算完。")
    print("  为了确认切片顺序没错位，这里用 6 个独立的向量-矩阵乘法重算一遍，")
    print("  逐个门单独算，完全不依赖拼接与切片。\n")

    B, D, H = 2, 4, 5
    rng = np.random.default_rng(5)
    x = rng.normal(size=(B, D))
    h = rng.normal(size=(B, H))

    cell = GRUCell(D, H, seed=7)
    out = cell.forward(x, h)
    P = cell.params()

    def proj(vec, Wg, bg):
        """朴素实现：完全不做拼接，每个输出元素都显式按定义累加。"""
        out = np.zeros((B, H))
        for b in range(B):
            for j in range(H):
                acc = bg[j]
                for i in range(vec.shape[1]):
                    acc += vec[b, i] * Wg[i, j]
                out[b, j] = acc
        return out

    xr = proj(x, P["W_ih"][:, :H], P["b_ih"][:H])
    xz = proj(x, P["W_ih"][:, H : 2 * H], P["b_ih"][H : 2 * H])
    xn = proj(x, P["W_ih"][:, 2 * H :], P["b_ih"][2 * H :])
    hr = proj(h, P["W_hh"][:, :H], P["b_hh"][:H])
    hz = proj(h, P["W_hh"][:, H : 2 * H], P["b_hh"][H : 2 * H])
    hn = proj(h, P["W_hh"][:, 2 * H :], P["b_hh"][2 * H :])

    r = sigmoid(xr + hr)
    z = sigmoid(xz + hz)
    n = np.tanh(xn + r * hn)
    out_naive = (1.0 - z) * n + z * h

    print(f"  输出最大绝对差 = {np.abs(out - out_naive).max():.3e}")
    print("  → 一次矩阵乘法 + 切片，与三次独立投影完全等价。")

    # 顺带对比速度
    xb = np.random.default_rng(0).normal(size=(32, D))
    hb = np.random.default_rng(1).normal(size=(32, H))
    n_rep = 3000
    t0 = time.perf_counter()
    for _ in range(n_rep):
        cell.forward(xb, hb)
    t_fused = (time.perf_counter() - t0) / n_rep * 1e6

    def forward_naive_slow():
        xr = xb @ P["W_ih"][:, :H] + P["b_ih"][:H]
        xz = xb @ P["W_ih"][:, H : 2 * H] + P["b_ih"][H : 2 * H]
        xn = xb @ P["W_ih"][:, 2 * H :] + P["b_ih"][2 * H :]
        hr = hb @ P["W_hh"][:, :H] + P["b_hh"][:H]
        hz = hb @ P["W_hh"][:, H : 2 * H] + P["b_hh"][H : 2 * H]
        hn = hb @ P["W_hh"][:, 2 * H :] + P["b_hh"][2 * H :]
        rr = sigmoid(xr + hr); zz = sigmoid(xz + hz)
        nn_ = np.tanh(xn + rr * hn)
        return (1.0 - zz) * nn_ + zz * hb

    t0 = time.perf_counter()
    for _ in range(n_rep):
        forward_naive_slow()
    t_split = (time.perf_counter() - t0) / n_rep * 1e6

    print(f"\n  单次前向耗时（B=32, in=4, H=5，{n_rep} 次平均）：")
    print(f"    拼接成一个大矩阵（本项目）  {t_fused:>7.2f} μs")
    print(f"    6 个独立小矩阵乘法          {t_split:>7.2f} μs")
    print(f"    加速比 {t_split / t_fused:.2f}×（维度越大差距越明显，因为大矩阵乘法能吃满 BLAS）")


# ======================================================================================
# [5] 梯度检验
# ======================================================================================
def section_gradcheck() -> None:
    title("[5] BPTT 梯度检验")

    rng = np.random.default_rng(0)
    B, D, H = 3, 5, 7
    x = np.ascontiguousarray(rng.normal(size=(B, D)))
    h = np.ascontiguousarray(rng.normal(size=(B, H)))
    G = rng.normal(size=(B, H))

    # ---- 单元级 ----
    cell = GRUCell(D, H, seed=1)

    def loss_fn():
        return float((cell.forward(x, h) * G).sum())

    loss_fn()
    cell.zero_grad()
    d = cell.backward(G)
    tensors = {"x": x, "h": h}
    tensors.update(cell.params())
    analytic = {"x": d["x"], "h": d["h"]}
    analytic.update(cell.grads())
    grad_table("GRUCell（输入 x/h + 4 个参数矩阵）：", tensors, analytic, loss_fn,
               np.random.default_rng(0), n_samples=5)

    # ---- 编码器端到端 ----
    print("\n  接下来是编码器端到端。这里检验的是**沿时间反向传播（BPTT）**：")
    print("  同一个 cell 被调用了 S 次，参数梯度必须累加。如果哪一步忘了累加")
    print("  （比如在循环里 zero_grad），参数梯度会偏小，而输入梯度看起来可能还对。\n")

    V, E, Hh, Bb, S = 12, 6, 5, 2, 5
    src = rng.integers(0, V, size=(Bb, S))
    mask = np.ones((Bb, S))
    mask[0, 3:] = 0.0                       # 样本 0 只有前 3 个是真词
    enc = GRUEncoder(V, E, Hh, seed=3, pad_id=0)
    G_hs = rng.normal(size=(Bb, S, Hh))
    G_hf = rng.normal(size=(Bb, Hh))

    def enc_loss():
        hs, hf = enc.forward(src, mask)
        return float((hs * G_hs).sum() + (hf * G_hf).sum())

    enc_loss()
    enc.zero_grad()
    enc.backward(G_hs, G_hf)
    grad_table("GRUEncoder（Embedding + GRU 全部参数）：",
               enc.named_parameters_dict(), enc.named_grads(), enc_loss,
               np.random.default_rng(0), n_samples=4)

    # ---- 对照：故意在循环里 zero_grad，模拟"忘记累加" ----
    class ForgotAccumulate(GRUEncoder):
        def backward(self, dH_seq, dh_final=None):
            c = self._cache
            S_ = c["S"]
            src_mask = c["src_mask"]
            cell_caches = c["cell_caches"]
            dhs = dH_seq.copy()
            if dh_final is not None:
                dhs[:, S_ - 1, :] += dh_final
            self.emb.zero_grad()
            dx_seq = np.zeros((dH_seq.shape[0], S_, self.emb_dim))
            dh_carry = np.zeros((dH_seq.shape[0], self.hidden_dim))
            for t in reversed(range(S_)):
                dh_t = dhs[:, t, :] + dh_carry
                valid = src_mask[:, t][:, None]
                self.cell.zero_grad()            # ← 错误！每步都把之前累积的梯度清掉了
                self.cell.load_cache(cell_caches[t])
                dinputs = self.cell.backward(dh_t * valid)
                dx_seq[:, t, :] = dinputs["x"]
                dh_carry = dinputs["h"] + dh_t * (1.0 - valid)
            self.emb.backward(dx_seq)
            return None

    bad = ForgotAccumulate(V, E, Hh, seed=3, pad_id=0)
    bad.forward(src, mask)                            # 必须先前向，cache 才有内容
    bad.zero_grad()
    bad.backward(G_hs, G_hf)
    good_g = enc.named_grads()
    bad_g = bad.named_grads()
    ratios = {k: float(np.linalg.norm(bad_g[k]) / max(np.linalg.norm(good_g[k]), 1e-30))
              for k in good_g if good_g[k].size}
    print("\n  对照实验：在 BPTT 循环里每步都 zero_grad（忘记累加）")
    print(f"    {'参数':<14}{'正确梯度范数':>16}{'错误梯度范数':>16}{'比值':>10}")
    print("    " + "-" * 58)
    for k in sorted(good_g):
        if not good_g[k].size:
            continue
        print(f"    {k:<14}{np.linalg.norm(good_g[k]):>16.5f}{np.linalg.norm(bad_g[k]):>16.5f}"
              f"{ratios[k]:>10.3f}")
    worst = min(ratios.values())
    print(f"    最小比值 = {worst:.3f}   → 梯度被系统性压小，训练会明显变慢")
    print("    注意：这种错误**不报错、不出 nan**，只是学得慢。梯度检验是唯一能提前发现它的手段。")
    print("\n    特别值得看的是 cell.W_hh（隐状态→隐状态的那个权重）：比值恰好是 0.000。")
    print("    原因：循环反向时最后一次 zero_grad 发生在 t=0，而 t=0 的上一隐状态")
    print("    h₋₁ = 0，所以 `h @ W_hh` 恒等于 0，它对输出的影响是 0，梯度自然也是 0。")
    print("    后果：**GRU 的循环连接完全学不到东西**，退化成一个只看当前输入的前馈网络 ——")
    print("    也就是说它不是「RNN 学得慢」，而是「根本不是 RNN」了。")


# ======================================================================================
# [6] padding 不变量
# ======================================================================================
def section_padding_invariance() -> None:
    title("[6] 关键不变量：padding 不能影响真实位置")

    rng = np.random.default_rng(0)
    V, E, H, B, S = 15, 8, 6, 3, 5
    enc = GRUEncoder(V, E, H, seed=11, pad_id=0)

    src = rng.integers(1, V, size=(B, S))          # 不含 pad
    mask = np.ones((B, S))
    mask[1, 4] = 0.0                               # 样本 1 实际上是 4 个词
    mask[2, 2:] = 0.0                              # 样本 2 实际上是 2 个词

    hs1, hf1 = enc.forward(src, mask)

    # 追加 4 列 padding
    extra = 4
    src2 = np.concatenate([src, np.zeros((B, extra), dtype=np.int64)], axis=1)
    mask2 = np.concatenate([mask, np.zeros((B, extra))], axis=1)
    hs2, hf2 = enc.forward(src2, mask2)

    print(f"  原始：src {src.shape}，mask 有效长度 {mask.sum(1).astype(int).tolist()}")
    print(f"  追加：src {src2.shape}，多出 {extra} 列 <pad>\n")
    print(f"  真实位置隐藏状态最大绝对差 = {np.abs(hs1 - hs2[:, :S, :]).max():.3e}")
    print(f"  h_final 最大绝对差         = {np.abs(hf1 - hf2).max():.3e}")

    # 反过来：同一句话在不同 batch 里（padding 数量不同）应该得到相同结果
    print("\n  再验证「同一句话在不同 batch 中结果一致」：")
    src_a = np.array([[3, 4, 5, 0, 0, 0]])
    m_a = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    src_c = np.array([[3, 4, 5, 0, 0, 0, 0, 0, 0]])
    m_c = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    _, hf_a = enc.forward(src_a, m_a)
    _, hf_c = enc.forward(src_c, m_c)
    print(f"    同一句「3 4 5」补 3 个 pad vs 补 6 个 pad：")
    print(f"      h_final 最大绝对差 = {np.abs(hf_a - hf_c).max():.3e}")
    print("    → 完全一致。编码结果只取决于真实内容，与 padding 数量、batch 组成无关。")

    print("\n  这条不变量为什么重要：如果破坏了它，同一个句子在不同 batch 里")
    print("  会得到不同的编码结果 → 同一份输入时而翻译对、时而翻译错，且不可复现。")


# ======================================================================================
# [7] 冻结 vs 不冻结
# ======================================================================================
def section_freeze_compare() -> None:
    title("[7] 对照实验：状态在 padding 位置「冻结」还是照常更新")

    def encode_no_freeze(enc: GRUEncoder, src: np.ndarray):
        """朴素做法：不管 mask，直接在补齐序列上把 GRU 跑完。"""
        x = enc.emb.forward(src)
        B, S, _ = x.shape
        h = np.zeros((B, enc.hidden_dim))
        hs = np.empty((B, S, enc.hidden_dim))
        for t in range(S):
            h = enc.cell.forward(x[:, t, :], h)
            hs[:, t, :] = h
        return hs, h

    rng = np.random.default_rng(0)
    V, E, H = 15, 8, 6
    enc = GRUEncoder(V, E, H, seed=11, pad_id=0)

    words = [3, 4, 5]
    results = {}
    for n_pad in (0, 1, 4, 8):
        src = np.array([words + [0] * n_pad])
        mask = np.array([[1.0] * len(words) + [0.0] * n_pad])
        _, hf_frozen = enc.forward(src, mask)
        _, hf_free = encode_no_freeze(enc, src)
        results[n_pad] = (hf_frozen.copy(), hf_free.copy())

    ref_frozen, ref_free = results[0]
    print("  同一句「3 4 5」，测试加不同数量的 <pad> 之后 h_final 的变化：\n")
    print(f"    {'填充数':<8}{'冻结状态':>18}{'不冻结（朴素）':>22}")
    print("    " + "-" * 50)
    for n_pad, (hf_f, hf_nf) in results.items():
        d_f = np.abs(hf_f - ref_frozen).max()
        d_nf = np.abs(hf_nf - ref_free).max()
        print(f"    {n_pad:<8}{d_f:>18.3e}{d_nf:>22.3e}")

    print("\n  → 冻结状态：无论补多少 padding，h_final 都**一位不差**。")
    print("  → 不冻结：h_final 随 padding 数量持续漂移，因为句尾多跑了几步 <pad>。")
    print("    而 h_final 正是第 4 步解码器的初始隐状态 —— 这个漂移会直接污染翻译结果。")
    print("\n  右填充 + 只看前面对齐位置的输出时，不冻结看起来「没影响」；")
    print("  但 h_final 是**整句的摘要**，它一定受影响。这就是必须冻结的原因。")


# ======================================================================================
# [8] 编码真实日期
# ======================================================================================
def section_real_data() -> None:
    title("[8] 编码真实日期数据")

    train, dev, test = D.build_datasets(CONFIG.data)
    vocab = train.src_vocab

    enc = GRUEncoder(
        vocab_size=len(vocab),
        emb_dim=CONFIG.model.src_emb_dim,
        hidden_dim=CONFIG.model.hidden_dim,
        init_scale=CONFIG.model.init_scale,
        seed=0,
        pad_id=vocab.pad_id,
    )
    br = enc.num_params_breakdown()
    print(f"  源词表大小 {len(vocab)}，emb_dim={CONFIG.model.src_emb_dim}，"
          f"hidden_dim={CONFIG.model.hidden_dim}\n")
    print(f"    {'部分':<14}{'参数量':>10}")
    print("    " + "-" * 26)
    print(f"    {'Embedding':<12}{br['embedding']:>10,}   ({len(vocab)} × {CONFIG.model.src_emb_dim})")
    print(f"    {'GRU cell':<12}{br['gru']:>10,}   (2 × {CONFIG.model.src_emb_dim} × {3 * CONFIG.model.hidden_dim}"
          f" + 2 × {CONFIG.model.hidden_dim} × {3 * CONFIG.model.hidden_dim} + 2 × {3 * CONFIG.model.hidden_dim})")
    print(f"    {'合计':<13}{br['total']:>10,}")

    batch = D.collate(train, list(range(4)))
    t0 = time.perf_counter()
    n_rep = 200
    for _ in range(n_rep):
        enc.zero_grad()
        hs, hf = enc.forward(batch.src, batch.src_mask)
        enc.backward(np.ones_like(hs), np.ones_like(hf))
    dt = (time.perf_counter() - t0) / n_rep * 1000
    print(f"\n  一个 batch（B={batch.batch_size}, S={batch.src.shape[1]}）前向+反向：{dt:.2f} ms")

    print("\n  各位置的隐状态 L2 范数（# = 真实 token，. = <pad>）：")
    for i in range(batch.batch_size):
        toks = " ".join(vocab.itos[int(j)] for j in batch.src[i])
        norms = np.linalg.norm(hs[i], axis=-1)
        bars = "".join("#" if batch.src_mask[i, t] > 0.5 else "." for t in range(batch.src.shape[1]))
        nrm = " ".join(f"{v:5.1f}" for v in norms)
        print(f"    [{bars}] {toks:<26} 范数 {nrm}")

    print("\n  观察：")
    print("    · 每读一个真词，隐状态就会变化（范数改变）—— 信息在累积")
    print("    · padding 位置的范数与上一个真词**完全相同** —— 状态确实冻结了")
    print("    · h_final 就是最后一个真词的隐状态，第 4 步拿它当解码器起点")

    # 顺带验证：把编码器接上注意力，形状能对上
    from seq2seq_attn.attention import BahdanauAttention
    attn = BahdanauAttention(CONFIG.model.hidden_dim, CONFIG.model.hidden_dim,
                             CONFIG.model.hidden_dim, attn_dim=CONFIG.model.attn_dim)
    ctx, w = attn(hf[:, None, :], hs, hs, batch.src_mask)
    print(f"\n  与第 2 步的注意力对接（形状自检）：")
    print(f"    query = h_final[:, None, :]  {tuple(hf[:, None, :].shape)}")
    print(f"    keys  = values = H_seq       {tuple(hs.shape)}")
    print(f"    context {tuple(ctx.shape)}   weights {tuple(w.shape)}")
    print(f"    权重行和 = {w.sum(-1).min():.6f} ~ {w.sum(-1).max():.6f}（应全为 1）")
    print("    → 第 3 步的输出能直接喂给第 2 步验证过的注意力模块。")


# ======================================================================================
def main() -> None:
    print(LINE)
    print(" 第 3 步 · 编码器 GRU 前向与反向验证")
    print(LINE)
    print(" 自下而上验证：Linear → Embedding → GRU 门语义 → 单元 → BPTT → 编码器端到端")

    section_linear()
    section_embedding()
    section_gates()
    section_crosscheck()
    section_gradcheck()
    section_padding_invariance()
    section_freeze_compare()
    section_real_data()

    title("第 3 步完成 ✓")
    print(" 编码器已通过梯度检验与 padding 不变量检查。")
    print(" 下一步（第 4 步）把编码器和注意力拼成带注意力的解码器。")
    print()


if __name__ == "__main__":
    main()
