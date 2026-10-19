"""「无注意力」基线与消融实验框架的断言测试。

运行：
    python tests/test_ablation.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
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
from seq2seq_attn.attention import (
    BahdanauAttention,
    DotProductAttention,
    MeanContextAttention,
    build_attention,
)
from seq2seq_attn.config import CONFIG
from seq2seq_attn.model import Seq2Seq
from seq2seq_attn.nn import check_gradient, summarize_gradient_report

PASSED = 0
CFG = CONFIG.model
ROOT = pathlib.Path(__file__).resolve().parents[1]


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        raise AssertionError(name)
    PASSED += 1


def close(a, b, tol=1e-12) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=0))


def load_ablation_module():
    """把 scripts/08_ablation.py 当模块加载（它不是包的一部分）。"""
    path = ROOT / "scripts" / "08_ablation.py"
    spec = importlib.util.spec_from_file_location("ablation_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
def test_mean_attention_shapes() -> None:
    print("\n== MeanContextAttention：形状与语义 ==")
    rng = np.random.default_rng(0)
    B, Tq, S, d = 3, 2, 5, 4
    q = rng.normal(size=(B, Tq, d))
    k = rng.normal(size=(B, S, d))
    v = rng.normal(size=(B, S, d))
    mask = np.ones((B, S))
    mask[1, 3:] = 0.0
    mask[2, 1:] = 0.0

    attn = MeanContextAttention(d, d, d)
    ctx, w = attn(q, k, v, mask)

    check("无参数", attn.num_params() == 0)
    check("context 形状 (B,Tq,dv)", ctx.shape == (B, Tq, d), str(ctx.shape))
    check("weights 形状 (B,Tq,S)", w.shape == (B, Tq, S), str(w.shape))
    check("每行和为 1", close(w.sum(-1), 1.0, tol=1e-12))
    check("padding 位置权重为 0",
          bool(np.all(w[np.broadcast_to(mask[:, None, :] < 0.5, w.shape)] == 0.0)))
    check("权重与 query 无关（同一个样本不同解码步相同）",
          all(close(w[b, 0], w[b, t]) for b in range(B) for t in range(Tq)))
    check("权重与 keys 无关（只由 mask 决定）",
          close(w[0, 0], np.array([0.2] * 5)))
    check("padding 样本的权重均匀落在有效位上",
          close(w[1, 0], np.array([1 / 3] * 3 + [0, 0])))

    # context 必须等于掩码平均
    for b in range(B):
        valid = mask[b] > 0.5
        manual = v[b][valid].mean(axis=0)
        check(f"样本 {b} 的 context = 有效位置的平均", close(ctx[b, 0], manual, tol=1e-12))

    # 与"均匀注意力"等价：手动实现一个权重恒为 mask/Σmask 的模型
    check("context 与 query 无关",
          close(MeanContextAttention(d, d, d)(q * 100, k, v, mask)[0], ctx, tol=1e-12))
    check("context 与 keys 无关",
          close(MeanContextAttention(d, d, d)(q, k * -7, v, mask)[0], ctx, tol=1e-12))


def test_mean_attention_gradient() -> None:
    print("\n== MeanContextAttention：反向传播 ==")
    rng = np.random.default_rng(1)
    B, Tq, S, d = 2, 3, 4, 5
    q = np.ascontiguousarray(rng.normal(size=(B, Tq, d)))
    k = np.ascontiguousarray(rng.normal(size=(B, S, d)))
    v = np.ascontiguousarray(rng.normal(size=(B, S, d)))
    mask = np.ones((B, S))
    mask[0, 2:] = 0.0

    attn = MeanContextAttention(d, d, d)
    G = rng.normal(size=(B, Tq, d))

    def loss_fn():
        ctx, _ = attn.forward(q, k, v, mask)
        return float((ctx * G).sum())

    loss_fn()
    attn.zero_grad()
    grads = attn.backward(G)
    check("返回三个输入梯度", set(grads) == {"query", "keys", "values"}, str(set(grads)))
    check("dquery 形状正确", grads["query"].shape == q.shape and close(grads["query"], 0.0))
    check("dkeys 形状正确", grads["keys"].shape == k.shape and close(grads["keys"], 0.0))

    res = check_gradient({"values": v}, {"values": grads["values"]}, loss_fn,
                         n_samples=10, eps=1e-5, rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("dvalues 通过梯度检验", s["passed"], f"最大分歧/抖动 {s['worst_ratio']:.2e}")

    # dvalues 的语义：每个有效位置 j 拿到 Σ_t (mask_j/Σmask) · G_t
    manual = np.zeros_like(v)
    for b in range(B):
        valid = mask[b] > 0.5
        manual[b][valid] = G[b].sum(axis=0) / valid.sum()
    check("dvalues 等于手算结果", close(grads["values"], manual, tol=1e-12))

    # 未 forward 就 backward
    try:
        MeanContextAttention(d, d, d).backward(G)
        check("未 forward 就 backward 报错", False)
    except RuntimeError:
        check("未 forward 就 backward 报错", True)

    # 形状校验仍然生效
    try:
        attn.forward(q, k, v[:, :2], mask)
        check("拒绝 keys/values 长度不一致", False)
    except ValueError:
        check("拒绝 keys/values 长度不一致", True)


def test_build_attention_factory() -> None:
    print("\n== 注意力工厂 ==")
    for attn_type, cls in (("bahdanau", BahdanauAttention),
                           ("dot", DotProductAttention),
                           ("mean", MeanContextAttention)):
        cfg = dataclasses.replace(CFG, attn_type=attn_type)
        m = build_attention(cfg, 16, 16, 16)
        check(f"attn_type={attn_type} -> {cls.__name__}", isinstance(m, cls))
    try:
        build_attention(dataclasses.replace(CFG, attn_type="nope"), 16, 16, 16)
        check("未知类型报错", False)
    except ValueError as exc:
        check("未知类型报错", "mean" in str(exc), "错误信息列出了可用选项")


def test_decoder_with_mean_attention() -> None:
    print("\n== 解码器换用无注意力基线 ==")
    train_set, _, test_set = D.build_datasets(CONFIG.data)
    cfg = dataclasses.replace(CFG, attn_type="mean")
    model = Seq2Seq.build(cfg, len(train_set.src_vocab), len(train_set.tgt_vocab),
                          train_set.src_vocab.pad_id, train_set.tgt_vocab.pad_id, seed=0)
    batch = D.collate(train_set, list(range(4)))

    check("注意力模块是 MeanContextAttention",
          isinstance(model.dec.attn, MeanContextAttention))
    check("注意力零参数", model.dec.attn.num_params() == 0)
    check("整模型参数少了注意力的那一部分",
          model.num_params() < T.build_model(train_set, CFG).num_params(),
          f"{model.num_params():,} vs {T.build_model(train_set, CFG).num_params():,}")

    loss, logits, weights = model.forward(batch)
    check("loss 有限", np.isfinite(loss))
    check("logits 形状正确",
          logits.shape == (4, batch.tgt_in.shape[1], len(train_set.tgt_vocab)))
    check("weights 形状正确",
          weights.shape == (4, batch.tgt_in.shape[1], batch.src.shape[1]))
    check("padding 权重为 0",
          bool(np.all(weights[np.broadcast_to(batch.src_mask[:, None, :] < 0.5,
                                              weights.shape)] == 0.0)))

    # 端到端反向：编码器仍然必须收到梯度（经 values 那条路）
    model.zero_grad()
    model.backward()
    pg = model.named_grads()
    enc_zero = [k for k, v in pg.items() if k.startswith("enc.") and v.size
                and float(np.linalg.norm(v)) == 0.0]
    check("编码器仍然收到梯度（经 dvalues 回传）", not enc_zero, str(enc_zero))
    attn_keys = [k for k in pg if k.startswith("dec.attn.")]
    check("注意力模块没有参数梯度", not attn_keys, str(attn_keys))

    # 端到端反向：用**真实的交叉熵损失**（带 tgt_mask）做梯度检验。
    #
    # ⚠ 这里必须用真实损失，不能用人造损失 `(logits * G).sum()`：
    #   人造损失在 padding 位置也有梯度，而 `Embedding(padding_idx=0)`
    #   会把 <pad> 行的梯度强制置零 —— 两者对不上，梯度检验会**误报失败**。
    #   真实的 CE 用 tgt_mask 把 padding 位置的 dlogits 置零，
    #   所以 <pad> 行本来就拿不到梯度，padding_idx 只是无副作用的双保险。
    small = D.collate(train_set, list(range(3)))          # 3 条长度不同 -> 一定有 padding
    check("测试批次里确实出现了 <pad>",
          bool((small.tgt_in == train_set.tgt_vocab.pad_id).any()),
          f"tgt_len={small.tgt_len.tolist()} width={small.tgt_in.shape[1]}")

    def loss_fn():
        return float(model.forward(small)[0])

    def analytic():
        model.zero_grad()
        model.forward(small)
        model.backward()
        return model.named_grads()

    loss_fn()
    ana = analytic()
    pad_grad = model.dec.emb.grads()["weight"][train_set.tgt_vocab.pad_id]
    check("真实损失下 <pad> 行的梯度本来就是 0", close(pad_grad, 0.0), str(pad_grad))

    res = check_gradient(model.named_parameters_dict(), ana, loss_fn,
                         n_samples=3, eps=1e-5, rng=np.random.default_rng(0))
    s = summarize_gradient_report(res)
    check("无注意力配置下整链路梯度检验通过（真实损失）", s["passed"],
          f"最大分歧/抖动 {s['worst_ratio']:.2e}")


def test_padding_idx_vs_unmasked_loss() -> None:
    print("\n== padding_idx 与「人造损失」的交互（一个容易误判的坑）==")
    print("  `Embedding(padding_idx=0)` 会把 <pad> 行的梯度强制置零。")
    print("  真实的交叉熵用 tgt_mask 已经把 padding 位置的 dlogits 置零了，")
    print("  所以两者一致；但用**未 mask 的人造损失**做梯度检验时，")
    print("  <pad> 行的真实梯度不是 0，就会和 analytic 对不上 —— 这不是 bug。")

    train_set, _, _ = D.build_datasets(CONFIG.data)
    cfg = dataclasses.replace(CFG, attn_type="mean")
    model = Seq2Seq.build(cfg, len(train_set.src_vocab), len(train_set.tgt_vocab),
                          train_set.src_vocab.pad_id, train_set.tgt_vocab.pad_id, seed=0)
    pad_id = train_set.tgt_vocab.pad_id
    batch = D.collate(train_set, list(range(3)))
    G = np.random.default_rng(0).normal(size=(3, batch.tgt_in.shape[1],
                                              len(train_set.tgt_vocab)))

    def unmasked_loss():
        hs, hf = model.enc.forward(batch.src, batch.src_mask)
        lg, _ = model.dec.forward(batch.tgt_in, hs, batch.src_mask, hf)
        return float((lg * G).sum())

    # 真实损失：<pad> 行梯度 = 0
    model.zero_grad()
    model.forward(batch)
    model.backward()
    real_pad = model.dec.emb.grads()["weight"][pad_id].copy()

    # 人造损失：<pad> 行梯度被 padding_idx 抹成 0，但数值梯度不是 0
    model.zero_grad()
    hs, hf = model.enc.forward(batch.src, batch.src_mask)
    lg, _ = model.dec.forward(batch.tgt_in, hs, batch.src_mask, hf)
    dd = model.dec.backward(G)
    unmasked_pad = model.dec.emb.grads()["weight"][pad_id].copy()

    from seq2seq_attn.nn import numerical_gradient_at
    unmasked_loss()
    par = model.dec.emb.params()["weight"]
    j = int(np.argmax(np.abs(unmasked_pad))) or 0
    num = numerical_gradient_at(unmasked_loss, par, [pad_id * par.shape[1] + j], 1e-5)
    check("真实损失下 <pad> 行梯度 = 0", close(real_pad, 0.0), str(real_pad[:3]))
    check("人造损失下 <pad> 行 analytic 也被 padding_idx 抹成 0",
          close(unmasked_pad, 0.0), str(unmasked_pad[:3]))
    check("但人造损失下 <pad> 行的数值梯度不是 0（所以会误报）",
          abs(float(num[0])) > 1e-6, f"{float(num[0]):.3e}")

    # 反向验证：把 padding_idx 关掉，人造损失的梯度检验就该通过了
    model.dec.emb.padding_idx = None
    model.zero_grad()
    hs, hf = model.enc.forward(batch.src, batch.src_mask)
    lg, _ = model.dec.forward(batch.tgt_in, hs, batch.src_mask, hf)
    model.dec.backward(G)
    from seq2seq_attn.nn import check_gradient as cg, summarize_gradient_report as sr
    res = cg(model.dec.emb.params(), model.dec.emb.grads(), unmasked_loss,
             n_samples=3, eps=1e-5, rng=np.random.default_rng(0))
    check("关掉 padding_idx 后人造损失的 Embedding 梯度检验通过",
          sr(res)["passed"], f"{sr(res)['worst_ratio']:.2e}")


def test_mean_attention_weights_are_uniform() -> None:
    print("\n== 无注意力基线的权重应当不携带对齐信息 ==")
    train_set, _, test_set = D.build_datasets(CONFIG.data)
    cfg = dataclasses.replace(CFG, attn_type="mean")
    model = Seq2Seq.build(cfg, len(train_set.src_vocab), len(train_set.tgt_vocab),
                          train_set.src_vocab.pad_id, train_set.tgt_vocab.pad_id, seed=0)
    batch = D.collate(test_set, list(range(4)))
    hs, hf = model.enc.forward(batch.src, batch.src_mask)
    _, weights = model.dec.forward(batch.tgt_in, hs, batch.src_mask, hf)

    # 权重只由 mask 决定：同一序列的所有解码步完全相同
    same = all(np.allclose(weights[i, 0], weights[i, t])
               for i in range(4) for t in range(weights.shape[1]))
    check("所有解码步的权重完全相同", same)
    # 且等于 mask 归一化
    expect = batch.src_mask / batch.src_mask.sum(axis=1, keepdims=True)
    check("权重 = mask 归一化",
          bool(np.allclose(weights[:, 0, :], expect, atol=1e-12)))


def test_ablation_harness() -> None:
    print("\n== 消融脚本的配置构造 ==")
    mod = load_ablation_module()

    check("配置表里有 baseline", "baseline" in mod.ABLATIONS)
    check("配置表里有 no_attention", "no_attention" in mod.ABLATIONS)
    check("每个配置都有说明", set(mod.ABLATIONS) <= set(mod.DESCRIPTIONS))
    check("baseline 不做任何覆盖", mod.ABLATIONS["baseline"] == {})
    check("每个配置都是单变量修改（只改 1 个字段）",
          all(len(v) == 1 for k, v in mod.ABLATIONS.items() if k != "baseline"),
          str({k: len(v) for k, v in mod.ABLATIONS.items()}))

    cfg = mod.make_config("no_attention", 500)
    check("make_config 改了 attn_type", cfg.model.attn_type == "mean")
    check("make_config 改了 n_train", cfg.data.n_train == 500)
    check("make_config 不动其他字段",
          cfg.data.n_test == CONFIG.data.n_test
          and cfg.train.lr == CONFIG.train.lr
          and cfg.model.hidden_dim == CONFIG.model.hidden_dim)
    check("基座配置未被就地修改", CONFIG.data.n_train == 4000 and
          CONFIG.model.attn_type == "bahdanau")

    # 500 条样本能正常构造数据集
    datasets = D.build_datasets(cfg.data)
    check("n_train=500 的数据集大小正确",
          len(datasets[0]) == 500 and len(datasets[1]) == 400 and len(datasets[2]) == 400,
          str([len(d) for d in datasets]))
    check("小训练集仍覆盖所有数字位",
          len({e.year for e in datasets[0].examples}) > 300,
          str(len({e.year for e in datasets[0].examples})))

    p = mod.result_path("baseline", 500)
    check("结果路径命名规则", p.name == "baseline_n500.json", p.name)


def main() -> None:
    print("=" * 70)
    print(" 无注意力基线与消融框架 测试")
    print("=" * 70)
    test_mean_attention_shapes()
    test_mean_attention_gradient()
    test_build_attention_factory()
    test_decoder_with_mean_attention()
    test_padding_idx_vs_unmasked_loss()
    test_mean_attention_weights_are_uniform()
    test_ablation_harness()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
