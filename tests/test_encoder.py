"""编码器（Linear / Embedding / GRUCell / GRUEncoder）的断言测试。

运行：
    python tests/test_encoder.py
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
from seq2seq_attn.config import CONFIG
from seq2seq_attn.encoder import GRUCell, GRUEncoder
from seq2seq_attn.nn import (
    Embedding,
    Linear,
    Module,
    check_gradient,
    sigmoid,
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


def report(tensors, analytic, loss_fn, n_samples=5, eps=1e-5, seed=0) -> dict:
    """跑一次梯度检验并返回汇总（**不抛异常**，供期望失败的回归测试用）。"""
    res = check_gradient(tensors, analytic, loss_fn, n_samples=n_samples, eps=eps,
                         rng=np.random.default_rng(seed))
    return summarize_gradient_report(res)


def gradcheck(tensors, analytic, loss_fn, n_samples=5, eps=1e-5, seed=0) -> float:
    """校验通过时返回「分歧/抖动」这个自校准判据的最差值（O(1)~O(100) 表示通过）。"""
    s = report(tensors, analytic, loss_fn, n_samples, eps, seed)
    if not s["passed"]:
        raise AssertionError(f"梯度检验不通过：{s['failed']}")
    return s["worst_ratio"]


# --------------------------------------------------------------------------------------
def test_sigmoid() -> None:
    print("\n== sigmoid ==")
    x = np.array([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    y = sigmoid(x)
    check("极端输入无 nan / inf", bool(np.isfinite(y).all()))
    check("极负下溢到精确 0（x=-1000）", y[0] == 0.0, str(y[0]))
    check("x=-50 时约 1.9e-22（不需要精确为 0）", y[1] < 1e-20 and y[1] > 0, str(y[1]))
    check("极正饱和到精确 1", y[-1] == 1.0 and y[5] == 1.0)
    check("sigmoid(0) = 0.5", abs(y[3] - 0.5) < 1e-15)
    naive = 1.0 / (1.0 + np.exp(-x[2:5]))
    check("安全区间与朴素写法一致", close(y[2:5], naive))
    check("值域在 [0,1]", bool(np.all((y >= 0) & (y <= 1))))

    # 导数：σ' = σ(1-σ)
    xs = np.ascontiguousarray(np.random.default_rng(0).normal(size=(2, 5)))
    G = np.random.default_rng(1).normal(size=(2, 5))
    eps = 1e-6
    num = np.zeros_like(xs)
    for idx in np.ndindex(xs.shape):
        old = xs[idx]
        xs[idx] = old + eps; fp = float((sigmoid(xs) * G).sum())
        xs[idx] = old - eps; fm = float((sigmoid(xs) * G).sum())
        xs[idx] = old
        num[idx] = (fp - fm) / (2 * eps)
    # 注意：sigmoid 没有 backward 方法，这里用 σ(1-σ) 直接验算
    ana = sigmoid(xs) * (1 - sigmoid(xs)) * G
    check("导数 σ(1-σ) 与数值梯度一致",
          float(np.max(np.abs(ana - num) / np.maximum(np.abs(ana) + np.abs(num), 1e-12))) < 1e-6)


def test_module_registry() -> None:
    print("\n== Module 参数登记与递归 ==")

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.lin = self.register_module("lin", Linear(4, 3, seed=0))
            self.sub = self.register_module("sub", Linear(3, 2, seed=1))

        def forward(self, x):
            return self.sub.forward(self.lin.forward(x))

    net = Net()
    names = sorted(n for n, _ in net.named_parameters())
    check("递归收集参数名带前缀", names == ["lin.W", "lin.b", "sub.W", "sub.b"], str(names))
    check("num_params 递归求和", net.num_params() == 4 * 3 + 3 + 3 * 2 + 2,
          f"{net.num_params()}")
    check("自己没参数", len(net.params()) == 0)
    check("named_grads 键与 named_parameters 一致",
          sorted(net.named_grads()) == names)

    x = np.ascontiguousarray(np.random.default_rng(0).normal(size=(2, 4)))
    net.forward(x)
    d = np.ones((2, 2))
    net.sub.backward(d)
    check("子模块收到梯度", bool(np.any(net.named_grads()["sub.W"] != 0)))
    net.zero_grad()
    check("zero_grad 递归清零", all(bool(np.all(g == 0)) for g in net.named_grads().values()))

    try:
        net.register_module("bad", 123)
        check("register_module 拒绝非 Module", False)
    except TypeError:
        check("register_module 拒绝非 Module", True)


def test_linear() -> None:
    print("\n== Linear ==")
    rng = np.random.default_rng(0)
    lin = Linear(5, 3, seed=1)
    x = np.ascontiguousarray(rng.normal(size=(2, 4, 5)))
    y = lin(x)
    check("形状 (..., in) -> (..., out)", y.shape == (2, 4, 3), str(y.shape))
    check("参数量 = in·out + out", lin.num_params() == 5 * 3 + 3)

    # 与手写二维矩阵乘法一致
    x2 = x.reshape(-1, 5)
    manual = (x2 @ lin.params()["W"] + lin.params()["b"]).reshape(2, 4, 3)
    check("与手算一致", close(y, manual))

    # 无偏置
    nb = Linear(5, 3, bias=False, seed=1)
    check("bias=False 时没有 b", set(nb.params()) == {"W"})

    # 梯度检验
    lin2 = Linear(5, 3, seed=2)
    xg = np.ascontiguousarray(rng.normal(size=(2, 4, 5)))
    G = rng.normal(size=(2, 4, 3))

    def loss_fn():
        return float((lin2.forward(xg) * G).sum())

    loss_fn()
    lin2.zero_grad()
    dx = lin2.backward(G)
    tensors = {"x": xg}; tensors.update(lin2.params())
    analytic = {"x": dx}; analytic.update(lin2.grads())
    err = gradcheck(tensors, analytic, loss_fn)
    check("参数与输入梯度全通过", err < 1e3, f"分歧/抖动 {err:.2e}")

    # 未 forward 就 backward
    try:
        Linear(2, 2, seed=0).backward(np.ones((1, 2)))
        check("未 forward 就 backward 报错", False)
    except RuntimeError:
        check("未 forward 就 backward 报错", True)


def test_embedding() -> None:
    print("\n== Embedding ==")
    emb = Embedding(6, 4, seed=0)
    check("参数量 = num_emb × dim", emb.num_params() == 24)
    idx = np.array([[1, 2, 1], [0, 5, 5]])
    out = emb(idx)
    check("输出形状 (...,) -> (..., D)", out.shape == (2, 3, 4), str(out.shape))
    check("查表正确", close(out[0, 0], emb.params()["weight"][1]))
    check("forward 返回的是拷贝（改不动权重表）",
          not np.shares_memory(out, emb.params()["weight"]))

    # ---- 回归测试：重复索引必须用 np.add.at ----
    emb2 = Embedding(3, 2, seed=0)
    emb2.params()["weight"][...] = np.arange(6).reshape(3, 2)
    idx2 = np.array([[1, 1, 1]])
    dout = np.ones((1, 3, 2))
    emb2.forward(idx2)
    emb2.zero_grad()
    emb2.backward(dout)
    g = emb2.grads()["weight"]
    check("重复索引梯度累加（id=1 出现 3 次 -> 梯度 3）", close(g[1], [3.0, 3.0]), str(g[1]))
    check("未出现的 id 梯度为 0", close(g[0], [0.0, 0.0]) and close(g[2], [0.0, 0.0]))

    # 反例：错误写法会丢掉重复项
    wrong = np.zeros_like(emb2.params()["weight"])
    wrong[idx2.reshape(-1)] += dout.reshape(-1, 2)
    check("反例确认错误写法会丢梯度", close(wrong[1], [1.0, 1.0]), str(wrong[1]))

    # 梯度检验
    emb3 = Embedding(5, 3, seed=3)
    idxg = np.array([[1, 1, 2, 1], [0, 3, 3, 4]])
    Gg = np.random.default_rng(1).normal(size=(2, 4, 3))

    def loss_fn():
        return float((emb3.forward(idxg) * Gg).sum())

    loss_fn()
    emb3.zero_grad()
    emb3.backward(Gg)
    err = gradcheck(dict(emb3.params()), dict(emb3.grads()), loss_fn)
    check("梯度检验（含重复索引）", err < 1e3, f"分歧/抖动 {err:.2e}")

    # padding_idx
    embp = Embedding(5, 3, seed=0, padding_idx=0)
    check("padding_idx 权重初始为 0", close(embp.params()["weight"][0], np.zeros(3)))
    embp.forward(np.array([[0, 0, 1]]))
    embp.backward(np.ones((1, 3, 3)))
    check("padding_idx 梯度被钉在 0", close(embp.grads()["weight"][0], np.zeros(3)))

    # 越界
    try:
        Embedding(3, 2, seed=0).forward(np.array([[5]]))
        check("越界 token id 报错", False)
    except IndexError:
        check("越界 token id 报错", True)
    try:
        Embedding(3, 2, padding_idx=9)
        check("非法 padding_idx 报错", False)
    except ValueError:
        check("非法 padding_idx 报错", True)


def test_gru_cell() -> None:
    print("\n== GRUCell ==")
    D_, H = 5, 4
    cell = GRUCell(D_, H, seed=1)
    check("4 个参数矩阵", set(cell.params()) == {"W_ih", "W_hh", "b_ih", "b_hh"})
    check("W_ih 形状 (in, 3H)", cell.params()["W_ih"].shape == (D_, 3 * H))
    check("W_hh 形状 (H, 3H)", cell.params()["W_hh"].shape == (H, 3 * H))
    check("参数量", cell.num_params() == D_ * 3 * H + H * 3 * H + 2 * 3 * H)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, D_))
    h = rng.normal(size=(3, H))
    out = cell(x, h)
    check("输出形状 (B, H)", out.shape == (3, H), str(out.shape))

    # ---- 门语义 ----
    def cell_with(bias_z=0.0, bias_r=0.0):
        c = GRUCell(D_, H, seed=1)
        c.params()["b_ih"][H:2 * H] = bias_z
        c.params()["b_hh"][H:2 * H] = bias_z
        c.params()["b_ih"][0:H] = bias_r
        c.params()["b_hh"][0:H] = bias_r
        return c

    z1 = cell_with(bias_z=30.0)
    o1 = z1(x, h)
    o2 = z1(x * 100.0, h)
    check("z≈1 时输出与输入无关（完全保留记忆）", close(o1, o2, tol=1e-12))
    check("z≈1 时 h' == h", close(o1, h, tol=1e-12))

    z0 = cell_with(bias_z=-30.0, bias_r=-30.0)
    a = z0(x, h)
    b = z0(x, h * -5.0)
    check("z≈0 且 r≈0 时输出与历史严格无关（重置记忆）", close(a, b, tol=1e-12))

    z0_only = cell_with(bias_z=-30.0)
    c1 = z0_only(x, h)
    c2 = z0_only(x, h * -5.0)
    check("z≈0 但 r 正常时输出仍受历史影响", not close(c1, c2, tol=1e-6))

    # ---- 与朴素逐门实现交叉验证 ----
    P = cell.params()
    B = 2
    xb = np.ascontiguousarray(rng.normal(size=(B, D_)))
    hb = np.ascontiguousarray(rng.normal(size=(B, H)))

    def naive_proj(vec, Wg, bg):
        out = np.zeros((vec.shape[0], Wg.shape[1]))
        for bi in range(vec.shape[0]):
            for j in range(Wg.shape[1]):
                acc = bg[j]
                for i in range(vec.shape[1]):
                    acc += vec[bi, i] * Wg[i, j]
                out[bi, j] = acc
        return out

    xr = naive_proj(xb, P["W_ih"][:, :H], P["b_ih"][:H])
    xz = naive_proj(xb, P["W_ih"][:, H:2 * H], P["b_ih"][H:2 * H])
    xn = naive_proj(xb, P["W_ih"][:, 2 * H:], P["b_ih"][2 * H:])
    hr = naive_proj(hb, P["W_hh"][:, :H], P["b_hh"][:H])
    hz = naive_proj(hb, P["W_hh"][:, H:2 * H], P["b_hh"][H:2 * H])
    hn = naive_proj(hb, P["W_hh"][:, 2 * H:], P["b_hh"][2 * H:])
    rr = sigmoid(xr + hr); zz = sigmoid(xz + hz)
    nn_ = np.tanh(xn + rr * hn)
    naive_out = (1 - zz) * nn_ + zz * hb
    check("与朴素逐门实现一致", close(cell(xb, hb), naive_out, tol=1e-12))

    # ---- 梯度检验 ----
    cell2 = GRUCell(D_, H, seed=2)
    xg = np.ascontiguousarray(rng.normal(size=(3, D_)))
    hg = np.ascontiguousarray(rng.normal(size=(3, H)))
    G = rng.normal(size=(3, H))

    def loss_fn():
        return float((cell2.forward(xg, hg) * G).sum())

    loss_fn()
    cell2.zero_grad()
    d = cell2.backward(G)
    tensors = {"x": xg, "h": hg}; tensors.update(cell2.params())
    analytic = {"x": d["x"], "h": d["h"]}; analytic.update(cell2.grads())
    err = gradcheck(tensors, analytic, loss_fn)
    check("输入 + 参数梯度全通过", err < 1e3, f"分歧/抖动 {err:.2e}")

    # 累加语义
    cell3 = GRUCell(D_, H, seed=3)
    cell3.zero_grad()
    cell3.forward(xg, hg); cell3.backward(G)
    g1 = {k: v.copy() for k, v in cell3.grads().items()}
    cell3.forward(xg, hg); cell3.backward(G)
    check("backward 两次梯度翻倍", all(close(cell3.grads()[k], 2 * g1[k]) for k in g1))


def test_encoder_shapes_and_padding() -> None:
    print("\n== GRUEncoder：形状与 padding 不变量 ==")
    V, E, H, B, S = 12, 6, 5, 3, 7
    rng = np.random.default_rng(0)
    enc = GRUEncoder(V, E, H, seed=0, pad_id=0)
    br = enc.num_params_breakdown()
    check("参数量拆解相加", br["embedding"] + br["gru"] == br["total"], str(br))
    check("named_parameters 前缀正确",
          sorted(n for n, _ in enc.named_parameters()) ==
          ["cell.W_hh", "cell.W_ih", "cell.b_hh", "cell.b_ih", "emb.weight"])

    src = rng.integers(1, V, size=(B, S))
    mask = np.ones((B, S))
    mask[1, 5:] = 0.0
    mask[2, 2:] = 0.0
    hs, hf = enc.forward(src, mask)
    check("H_seq 形状 (B,S,H)", hs.shape == (B, S, H), str(hs.shape))
    check("h_final 形状 (B,H)", hf.shape == (B, H), str(hf.shape))
    check("h_final == H_seq 最后一位", close(hf, hs[:, S - 1, :]))
    check("h_final 是拷贝（外部改不到内部）", not np.shares_memory(hf, hs))

    # padding 位置状态冻结
    frozen_ok = True
    for b in range(B):
        real = int(mask[b].sum())
        for t in range(real, S):
            if not close(hs[b, t], hs[b, real - 1]):
                frozen_ok = False
    check("padding 位置的状态与最后一个真词完全相同", frozen_ok)

    # ---- 核心不变量：追加 padding 不改变任何东西 ----
    extra = 4
    src2 = np.concatenate([src, np.zeros((B, extra), dtype=np.int64)], axis=1)
    mask2 = np.concatenate([mask, np.zeros((B, extra))], axis=1)
    hs2, hf2 = enc.forward(src2, mask2)
    check("追加 padding 后真实位置输出逐位不变", close(hs, hs2[:, :S, :]))
    check("追加 padding 后 h_final 逐位不变", close(hf, hf2))

    # ---- 同一句子在不同 batch 组成下结果一致 ----
    a = np.array([[3, 4, 5, 0, 0]])
    ma = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
    c = np.array([[3, 4, 5, 0, 0, 0, 0, 0]])
    mc = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    hs_a, hfa = enc.forward(a, ma)
    hsc, hfc = enc.forward(c, mc)
    check("h_final 与 padding 数量无关", close(hfa, hfc))
    check("真实位置 H_seq 与 padding 数量无关", close(hs_a, hsc[:, :5, :]))

    # ---- 形状校验 ----
    for label, bad in [("src 不是二维", (np.zeros(3, dtype=np.int64), np.zeros(3))),
                       ("mask 形状不匹配", (src, np.ones((B, S + 1))))]:
        try:
            enc.forward(*bad)
            check(f"拒绝：{label}", False)
        except ValueError:
            check(f"拒绝：{label}", True)

    # ---- 无 padding 时等价于朴素不冻结实现（这是唯一两者应当一致的场合）----
    no_pad_src = rng.integers(1, V, size=(2, 4))
    no_pad_mask = np.ones((2, 4))
    hs_np, hf_np = enc.forward(no_pad_src, no_pad_mask)
    x = enc.emb.forward(no_pad_src)
    h = np.zeros((2, H))
    same = True
    for t in range(4):
        h = enc.cell.forward(x[:, t, :], h)
        if not close(h, hs_np[:, t, :]):
            same = False
    check("无 padding 时与朴素逐帧实现一致", same)
    check("无 padding 时 h_final 也一致", close(h, hf_np))


def test_encoder_gradients() -> None:
    print("\n== GRUEncoder：BPTT 梯度检验 ==")
    rng = np.random.default_rng(0)
    V, E, H, B, S = 10, 5, 4, 2, 5
    enc = GRUEncoder(V, E, H, seed=3, pad_id=0)
    src = rng.integers(0, V, size=(B, S))
    mask = np.ones((B, S))
    mask[0, 3:] = 0.0
    G_hs = rng.normal(size=(B, S, H))
    G_hf = rng.normal(size=(B, H))

    def loss_fn():
        hs, hf = enc.forward(src, mask)
        return float((hs * G_hs).sum() + (hf * G_hf).sum())

    loss_fn()
    enc.zero_grad()
    enc.backward(G_hs, G_hf)
    err = gradcheck(enc.named_parameters_dict(), enc.named_grads(), loss_fn,
                    n_samples=4, seed=1)
    check("全部参数梯度通过", err < 1e3, f"分歧/抖动 {err:.2e}")

    # dh_final=None 应该等价于把它加到最后一列
    enc.zero_grad(); enc.backward(G_hs, G_hf)
    g_with = {k: v.copy() for k, v in enc.named_grads().items()}
    enc.zero_grad(); enc.backward(G_hs + np.concatenate(
        [np.zeros((B, S - 1, H)), G_hf[:, None, :]], axis=1), None)
    g_wo = enc.named_grads()
    check("dh_final 与加到 dH_seq[:,-1] 等价",
          all(close(g_with[k], g_wo[k]) for k in g_with))

    # padding 位置的输出梯度会沿"冻结"那条路径原样回传到上一步（不该被丢掉）
    G_pad = G_hs.copy()
    G_pad[0, 3:, :] = 123.0                     # 在 padding 位置上加巨大梯度
    enc.zero_grad(); enc.backward(G_hs, G_hf)
    g_clean = {k: v.copy() for k, v in enc.named_grads().items()}
    enc.zero_grad(); enc.backward(G_pad, G_hf)
    g_dirty = enc.named_grads()
    check("padding 位置输出上的梯度会经冻结路径回传，不被丢弃",
          not all(close(g_clean[k], g_dirty[k]) for k in g_clean))

    # 未 forward 就 backward
    try:
        GRUEncoder(V, E, H, seed=0).backward(G_hs)
        check("未 forward 就 backward 报错", False)
    except RuntimeError:
        check("未 forward 就 backward 报错", True)

    # dH_seq 形状错误
    enc.forward(src, mask)
    try:
        enc.backward(np.zeros((B, S + 1, H)))
        check("dH_seq 形状错误时报错", False)
    except ValueError:
        check("dH_seq 形状错误时报错", True)


def test_cache_restore_regression() -> None:
    print("\n== 回归测试：BPTT 必须逐步恢复中间量 ==")
    print("  （这是本项目第 3 步真的踩到的 bug：cell._cache 只有一个槽位，")
    print("    前向循环调用 S 次后只剩最后一次，反向时每一步都用错数据。）")

    class NoCacheRestore(GRUEncoder):
        """故意不恢复中间量，模拟这个 bug。"""

        def backward(self, dH_seq, dh_final=None):
            c = self._cache
            S_ = c["S"]
            src_mask = c["src_mask"]
            dhs = dH_seq.copy()
            if dh_final is not None:
                dhs[:, S_ - 1, :] += dh_final
            self.cell.zero_grad()
            self.emb.zero_grad()
            dx_seq = np.zeros((dH_seq.shape[0], S_, self.emb_dim))
            dh_carry = np.zeros((dH_seq.shape[0], self.hidden_dim))
            for t in reversed(range(S_)):
                dh_t = dhs[:, t, :] + dh_carry
                valid = src_mask[:, t][:, None]
                # ← 少了 self.cell.load_cache(c["cell_caches"][t])
                dinputs = self.cell.backward(dh_t * valid)
                dx_seq[:, t, :] = dinputs["x"]
                dh_carry = dinputs["h"] + dh_t * (1.0 - valid)
            self.emb.backward(dx_seq)
            return None

    rng = np.random.default_rng(0)
    V, E, H, B, S = 10, 5, 4, 2, 5
    src = rng.integers(0, V, size=(B, S))
    mask = np.ones((B, S))
    mask[0, 3:] = 0.0
    G_hs = rng.normal(size=(B, S, H))
    G_hf = rng.normal(size=(B, H))

    def make_loss(enc):
        def loss_fn():
            hs, hf = enc.forward(src, mask)
            return float((hs * G_hs).sum() + (hf * G_hf).sum())
        return loss_fn

    good = GRUEncoder(V, E, H, seed=3, pad_id=0)
    lf = make_loss(good)
    lf(); good.zero_grad(); good.backward(G_hs, G_hf)
    s_good = report(good.named_parameters_dict(), good.named_grads(), lf, n_samples=4, seed=1)

    bad = NoCacheRestore(V, E, H, seed=3, pad_id=0)
    lf2 = make_loss(bad)
    lf2(); bad.zero_grad(); bad.backward(G_hs, G_hf)
    s_bad = report(bad.named_parameters_dict(), bad.named_grads(), lf2, n_samples=4, seed=1)

    check("正确实现通过", s_good["passed"], f"分歧/抖动 {s_good['worst_ratio']:.2e}")
    check("不恢复中间量的实现被抓住", not s_bad["passed"], f"不通过项 {s_bad['failed'][:2]}")
    check("两者相差 5 个数量级以上",
          s_bad["worst_ratio"] / max(s_good["worst_ratio"], 1e-30) > 1e5,
          f"{s_bad['worst_ratio']:.2e} vs {s_good['worst_ratio']:.2e}")

    # save_cache / load_cache 本身
    cell = GRUCell(3, 2, seed=0)
    x1 = np.array([[1.0, 2.0, 3.0]]); h1 = np.zeros((1, 2))
    cell.forward(x1, h1)
    snap = cell.save_cache()
    x2 = np.array([[9.0, 9.0, 9.0]])
    cell.forward(x2, h1)
    check("save_cache 返回的是快照（浅拷贝）", snap is not cell._cache)
    cell.load_cache(snap)
    check("load_cache 恢复后是原来那一份", close(cell._cache["x"], x1))
    check("快照未被后续 forward 污染", close(snap["x"], x1))
    fresh = GRUCell(3, 2, seed=0)
    check("未 forward 时 save_cache 返回 None", fresh.save_cache() is None)


def test_real_data_integration() -> None:
    print("\n== 真实数据 + 与注意力对接 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    vocab = train.src_vocab
    enc = GRUEncoder(len(vocab), CONFIG.model.src_emb_dim, CONFIG.model.hidden_dim,
                     init_scale=CONFIG.model.init_scale, seed=0, pad_id=vocab.pad_id)
    batch = D.collate(train, list(range(8)))
    hs, hf = enc.forward(batch.src, batch.src_mask)
    check("批大小与序列长度正确",
          hs.shape == (8, batch.src.shape[1], CONFIG.model.hidden_dim), str(hs.shape))
    check("输出无 nan", bool(np.isfinite(hs).all() and np.isfinite(hf).all()))

    # padding 位置冻结（真实数据里有 padding）
    ok = True
    for b in range(8):
        real = int(batch.src_len[b])
        for t in range(real, batch.src.shape[1]):
            if not close(hs[b, t], hs[b, real - 1], tol=1e-13):
                ok = False
    check("真实 batch 中 padding 位置状态冻结", ok)

    # 端到端梯度检验（小配置，速度可控）
    small = GRUEncoder(len(vocab), 8, 6, seed=1, pad_id=vocab.pad_id)
    small_src = batch.src[:2, :6]
    small_mask = batch.src_mask[:2, :6]
    G = np.random.default_rng(0).normal(size=(2, 6, 6))
    Gf = np.random.default_rng(1).normal(size=(2, 6))

    def loss_fn():
        a, b_ = small.forward(small_src, small_mask)
        return float((a * G).sum() + (b_ * Gf).sum())

    loss_fn()
    small.zero_grad()
    small.backward(G, Gf)
    err = gradcheck(small.named_parameters_dict(), small.named_grads(), loss_fn,
                    n_samples=3, seed=2)
    check("真实词表下的梯度检验", err < 1e3, f"分歧/抖动 {err:.2e}")

    # 与第 2 步的注意力对接
    from seq2seq_attn.attention import BahdanauAttention, DotProductAttention
    for label, attn in [("加性",
                         BahdanauAttention(CONFIG.model.hidden_dim, CONFIG.model.hidden_dim,
                                           CONFIG.model.hidden_dim,
                                           attn_dim=CONFIG.model.attn_dim)),
                        ("点积", DotProductAttention(CONFIG.model.hidden_dim,
                                                     CONFIG.model.hidden_dim,
                                                     CONFIG.model.hidden_dim))]:
        ctx, w = attn(hf[:, None, :], hs, hs, batch.src_mask)
        check(f"{label}注意力对接：context 形状", ctx.shape == (8, 1, CONFIG.model.hidden_dim))
        check(f"{label}注意力对接：权重形状", w.shape == (8, 1, batch.src.shape[1]))
        check(f"{label}注意力对接：权重行和为 1", close(w.sum(-1), 1.0, tol=1e-12))
        padded_zero = True
        for b in range(8):
            if not np.all(w[b, 0][batch.src_mask[b] < 0.5] == 0.0):
                padded_zero = False
        check(f"{label}注意力对接：padding 权重为 0", padded_zero)


def main() -> None:
    print("=" * 70)
    print(" 编码器测试（Linear / Embedding / GRUCell / GRUEncoder）")
    print("=" * 70)
    test_sigmoid()
    test_module_registry()
    test_linear()
    test_embedding()
    test_gru_cell()
    test_encoder_shapes_and_padding()
    test_encoder_gradients()
    test_cache_restore_regression()
    test_real_data_integration()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
