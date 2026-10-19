"""第 5 步：交叉熵 + Adam + warmup/余弦退火 的完整训练。

运行：
    python scripts/05_train.py              # 完整训练（约 4~5 分钟）
    python scripts/05_train.py --quick      # 快速版（400 步，约 1 分钟）
    python scripts/05_train.py --steps 1500 # 自定步数
    python scripts/05_train.py --no-sanity  # 跳过健全性检查

流程：
    [1] 健全性检查：先过拟合 10 条样本（最省时间的排错手段）
    [2] 正式训练
    [3] loss 曲线
    [4] 学习率与梯度范数
    [5] 注意力集中度分析（有没有「注意力坍塌」）
    [6] 测试集样例（年份训练时从未出现）
    [7] teacher forcing vs 自回归：exposure bias
    [8] 保存 checkpoint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from seq2seq_attn import train as T
from seq2seq_attn.config import CONFIG

LINE = "=" * 82
SUB = "-" * 82
ROOT = pathlib.Path(__file__).resolve().parents[1]


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def bar(value: float, vmax: float, width: int = 40) -> str:
    if vmax <= 0:
        return ""
    n = int(round(value / vmax * width))
    return "█" * max(0, min(n, width))


def plot_series(steps, values, label: str, width: int = 46, n_bins: int = 28,
                fmt: str = "{:.4f}") -> None:
    """把一条曲线压成 n_bins 个桶，画成横向条形图。"""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if steps.size == 0:
        print(f"    {label}: （无数据）")
        return
    edges = np.linspace(0, steps.size, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        if lo >= steps.size:
            break
        rows.append((steps[lo:hi].mean(), float(values[lo:hi].mean())))
    vmax = max(v for _, v in rows)
    vmin = min(v for _, v in rows)
    print(f"    {label}   (min {fmt.format(vmin)}  max {fmt.format(vmax)})")
    for s, v in rows:
        print(f"      {int(s):>5} {bar(v, vmax, width):<{width}} {fmt.format(v)}")
    print()


# ======================================================================================
# [5] 注意力集中度
# ======================================================================================
def attention_stats(model, dataset, max_batches: int = 8, batch_size: int = 64) -> dict:
    """统计注意力权重的归一化熵。

        归一化熵 = H(weights) / log(有效 key 数)

        接近 1  -> 权重接近均匀分布（「注意力坍塌」：模型没在选，只是平均）
        接近 0  -> 权重集中在少数位置（真的在对齐）
        0.3~0.7 -> 健康的软对齐
    """
    batches = D.make_batches(dataset, batch_size, shuffle=False)[:max_batches]
    entries = []
    for batch in batches:
        h_seq, h_final = model.enc.forward(batch.src, batch.src_mask)
        _, weights = model.dec.forward(batch.tgt_in, h_seq, batch.src_mask, h_final)
        p = np.clip(weights, 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=-1)                    # (B,T)
        n_valid = batch.src_mask.sum(axis=-1)                      # (B,)
        norm = np.log(np.maximum(n_valid, 2.0))[:, None]
        norm_ent = entropy / norm
        valid = batch.tgt_mask > 0.5
        entries.append(norm_ent[valid])
    flat = np.concatenate(entries) if entries else np.array([np.nan])
    return {
        "mean": float(np.nanmean(flat)),
        "median": float(np.nanmedian(flat)),
        "p10": float(np.nanpercentile(flat, 10)),
        "p90": float(np.nanpercentile(flat, 90)),
    }


# ======================================================================================
# [6] 背诵检测
# ======================================================================================
def year_memorization_stats(model, train_set, test_set, max_len: int, n: int = 200) -> dict:
    """检测模型是不是在**背诵训练年份**而不是照抄源句。

    判据：统计自回归生成结果里的「年份部分」
        · 落在训练年份集合中的比例   —— 越接近 1 越可疑（背诵特征）
        · 等于参考年份的比例         —— 真正的正确率

    如果第一个比例很高而第二个很低，说明模型把训练年份背了下来，
    遇到没见过的年份就"猜一个见过的"。
    """
    rows = T.decode_sentences(model, test_set, range(min(n, len(test_set))), max_len)
    train_years = {str(e.year) for e in train_set.examples}
    test_years = {e.year for e in test_set.examples}

    gen_years = []
    for r in rows:
        parts = r["greedy"].split("/")
        gen_years.append(parts[-1] if len(parts) >= 2 else "")

    in_train = sum(1 for y in gen_years if y in train_years)
    correct = sum(1 for r in rows if r["ref"] == r["greedy"])
    total = max(len(rows), 1)
    return {
        "n": len(rows),
        "generated_year_in_train_set": in_train / total,
        "generated_year_correct": correct / total,
        "example_generated_years": sorted({y for y in gen_years if y})[:10],
        "example_test_years": sorted(test_years)[:10],
    }


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="第 5 步训练脚本")
    ap.add_argument("--steps", type=int, default=None, help="训练步数（默认取配置的 3000）")
    ap.add_argument("--quick", action="store_true", help="快速模式：400 步")
    ap.add_argument("--no-sanity", action="store_true", help="跳过健全性检查")
    ap.add_argument("--sanity-steps", type=int, default=400)
    ap.add_argument("--save", action="store_true", default=True)
    ap.add_argument("--no-save", dest="save", action="store_false")
    args = ap.parse_args()

    steps = 400 if args.quick else (args.steps or CONFIG.train.max_steps)
    eval_every = max(50, steps // 12)

    print(LINE)
    print(" 第 5 步 · 交叉熵 + Adam + warmup/余弦退火 的完整训练")
    print(LINE)
    print(f" 任务：中文日期 → 英文日期")
    print(f" 超参数：lr={CONFIG.train.lr}  batch={CONFIG.data.batch_size}  "
          f"clip={CONFIG.train.grad_clip}  steps={steps}")
    print(f" 模型：emb={CONFIG.model.src_emb_dim}  hidden={CONFIG.model.hidden_dim}  "
          f"attn={CONFIG.model.attn_type}  "
          f"decoder={CONFIG.model.decoder_output}/{CONFIG.model.decoder_init}")

    # ---------------------------------------------------------------- [1]
    sanity = None
    if not args.no_sanity:
        title("[1] 健全性检查：先过拟合 10 条样本")
        print("  如果连 10 条样本都记不住，那一定是代码有 bug，")
        print("  而不是「模型太小」「数据太难」—— 这时候调超参数纯属浪费时间。\n")
        sanity = T.sanity_check_overfit(CONFIG, n_examples=10, steps=args.sanity_steps)
        ok = sanity["final"]["seq_acc"] >= 1.0 and sanity["greedy"]["seq_acc"] >= 1.0
        print(f"\n  10 条样本：loss {sanity['first_loss']:.4f} → {sanity['final_loss']:.6f}，"
              f"用时 {sanity['elapsed']:.1f}s")
        print(f"  teacher forcing：token {sanity['final']['token_acc']:.4f}  "
              f"整句 {sanity['final']['seq_acc']:.4f}")
        print(f"  自回归贪心    ：token {sanity['greedy']['token_acc']:.4f}  "
              f"整句 {sanity['greedy']['seq_acc']:.4f}")
        print(f"\n  {'✓ 通过：前向、反向、优化器三件事都是通的' if ok else '✗ 没记住 10 条样本，先别往下走'}")

    # ---------------------------------------------------------------- [2]
    title("[2] 正式训练")
    print("  每个 step 的顺序：zero_grad → forward → backward → 记录梯度范数 → 裁剪 → step")
    print("  注意梯度范数在**裁剪之前**记录，否则永远看不到爆炸。\n")
    t0 = time.perf_counter()
    result = T.train(CONFIG, max_steps=steps, eval_every=eval_every,
                     log_every=max(50, steps // 15))
    total = time.perf_counter() - t0
    hist = result["history"]
    print(f"\n  整个脚本训练部分用时 {total:.1f}s")

    # ---------------------------------------------------------------- [3]
    title("[3] Loss 曲线")
    plot_series(hist["step"], hist["loss"], "训练 loss（batch 平均，已按桶平滑）")
    plot_series(hist["eval_step"], hist["dev_loss"], "开发集 loss",
                fmt="{:.4f}")
    n_eval = len(hist["eval_step"])
    print(f"    {'step':>6}{'dev loss':>12}{'TF token':>12}{'TF seq':>11}"
          f"{'AR token':>12}{'AR seq':>11}")
    print("    " + "-" * 64)
    for i in range(n_eval):
        print(f"    {hist['eval_step'][i]:>6}{hist['dev_loss'][i]:>12.4f}"
              f"{hist['dev_tf_token'][i]:>12.4f}{hist['dev_tf_seq'][i]:>11.4f}"
              f"{hist['dev_ar_token'][i]:>12.4f}{hist['dev_ar_seq'][i]:>11.4f}")

    # ---------------------------------------------------------------- [4]
    title("[4] 学习率与梯度范数")
    plot_series(hist["step"], hist["lr"], "学习率（warmup 10% + 余弦退火）",
                fmt="{:.3e}")
    gn = np.asarray(hist["grad_norm"], dtype=float)
    print(f"    梯度范数：min {gn.min():.3f}  median {np.median(gn):.3f}  "
          f"max {gn.max():.3f}   clip 阈值 {CONFIG.train.grad_clip}")
    n_clipped = int((gn > CONFIG.train.grad_clip).sum())
    print(f"    被裁剪的 step 数：{n_clipped} / {len(gn)}"
          f"（{n_clipped / max(len(gn), 1) * 100:.1f}%）")
    if n_clipped == 0:
        print("    → 一次都没触发裁剪，说明梯度很稳（序列只有 10 步，BPTT 不会爆炸）")
    else:
        print("    → 前几十步通常会有裁剪（初始化梯度大），后期应迅速归零")

    # ---------------------------------------------------------------- [5]
    title("[5] 注意力集中度：模型真的在「选」位置吗？")
    print("  归一化熵 = H(权重) / log(有效 key 数)")
    print("    接近 1 → 权重接近均匀（「注意力坍塌」，模型没在选，只是在平均）")
    print("    接近 0 → 权重集中在一两个位置（真的在对齐）\n")
    for name, ds in (("dev", result["dev_set"]), ("test", result["test_set"])):
        st = attention_stats(result["model"], ds)
        print(f"    {name:<6} 归一化熵  mean {st['mean']:.4f}  median {st['median']:.4f}"
              f"   p10 {st['p10']:.4f}  p90 {st['p90']:.4f}")
    print("\n  （第 7 步会把这些权重画成对齐热力图，直接看它有没有学对。）")

    # ---------------------------------------------------------------- [6]
    title(f"[6] 测试集样例（年份 {CONFIG.data.test_years[0]}~{CONFIG.data.test_years[1]}，"
          f"训练时从未出现）")
    rows = T.decode_sentences(result["model"], result["test_set"], range(14),
                              result["max_len"])
    print(f"    {'源句（中文）':<18}{'参考':<18}{'teacher forcing':<18}{'自回归贪心':<18}结果")
    print("    " + "-" * 82)
    n_ok = 0
    for r in rows:
        ok = r["ref"] == r["greedy"]
        n_ok += ok
        print(f"    {r['src']:<16}{r['ref']:<18}{r['tf']:<18}{r['greedy']:<18}"
              f"{'✓' if ok else '✗'}")
    print(f"\n    这 14 条里自回归全对 {n_ok} 条")

    # 挑几条错的看看错在哪
    all_rows = T.decode_sentences(result["model"], result["test_set"], range(120),
                                  result["max_len"])
    bad = [r for r in all_rows if r["ref"] != r["greedy"]]
    print(f"\n    前 120 条测试样本里自回归错误 {len(bad)} 条")
    if bad:
        print("    错误样例：")
        for r in bad[:8]:
            print(f"      {r['src']:<16} 参考 {r['ref']:<18} 贪心 {r['greedy']}")

    # ---- 背诵检测 ----
    print("\n  【背诵检测】模型生成的年份有多少落在**训练年份集合**里？")
    st = year_memorization_stats(result["model"], result["train_set"],
                                 result["test_set"], result["max_len"])
    print(f"    生成的年份属于训练年份集合的比例 = {st['generated_year_in_train_set']:.2%}")
    print(f"    生成的年份完全正确的比例         = {st['generated_year_correct']:.2%}")
    print(f"    模型实际生成过的年份（前 10 个）：{st['example_generated_years']}")
    print(f"    测试集真实年份（前 10 个）      ：{st['example_test_years']}")
    if st["generated_year_in_train_set"] > 0.9 and st["generated_year_correct"] < 0.3:
        print("\n    ✗ 典型的「背诵」特征：模型只会吐出训练时见过的年份，")
        print("      遇到没见过的年份就猜一个见过的 —— 它没学会照抄源句数字。")
        print("      这通常说明训练集的年份太集中（样本数 / 不同年份数 太大），")
        print("      或者该数位上的数字在训练时从未出现过。")
    elif st["generated_year_correct"] > 0.8:
        print("\n    ✓ 模型在生成**从没见过的年份**，说明它学到的是「照抄数字」这条规则，")
        print("      而不是把训练年份背下来。这才是真正意义上的组合泛化。")

    # ---------------------------------------------------------------- [7]
    title("[7] teacher forcing vs 自回归：exposure bias")
    tf_d = result["dev_tf"]["seq_acc"]
    ar_d = result["dev_ar"]["seq_acc"]
    tf_t = result["test_tf"]["seq_acc"]
    ar_t = result["test_ar"]["seq_acc"]
    print(f"    {'数据集':<8}{'TF 整句准确率':>16}{'AR 整句准确率':>16}{'差距':>10}")
    print("    " + "-" * 52)
    for name, tf, ar in (("dev", tf_d, ar_d), ("test", tf_t, ar_t)):
        print(f"    {name:<8}{tf:>16.4f}{ar:>16.4f}{tf - ar:>+10.4f}")
    print("\n  训练时第 t 步永远喂**真实的** y_{t-1}，推理时只能喂**自己生成的** ŷ_{t-1}。")
    print("  一旦某步出错，后面的输入就偏离了训练分布，误差会累积 —— 这就是 exposure bias。")

    # ---------------------------------------------------------------- [8]
    if args.save:
        title("[8] 保存 checkpoint")
        ckpt_dir = ROOT / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        name = "quick" if args.quick else f"steps{steps}"
        npz_path = ckpt_dir / f"{name}.npz"
        meta_path = ckpt_dir / f"{name}.json"
        np.savez_compressed(npz_path, **result["model"].state_dict())
        meta = {
            "run_name": name,
            "steps": steps,
            "data_config": dataclasses.asdict(CONFIG.data),
            "model_config": dataclasses.asdict(CONFIG.model),
            "train_config": dataclasses.asdict(CONFIG.train),
            "src_itos": result["train_set"].src_vocab.itos,
            "tgt_itos": result["train_set"].tgt_vocab.itos,
            "max_len": result["max_len"],
            "metrics": {
                k: result[k] for k in
                ("train_tf", "train_ar", "dev_tf", "dev_ar", "test_tf", "test_ar")
                if k in result
            },
            "elapsed": result["elapsed"],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        size_mb = npz_path.stat().st_size / 1024 / 1024
        print(f"    权重  {npz_path.relative_to(ROOT)}   ({size_mb:.2f} MB, "
              f"{result['model'].num_params():,} 个参数)")
        print(f"    元信息 {meta_path.relative_to(ROOT)}")
        print("    （第 6 步的 Beam Search 和第 7 步的热力图会直接读这个文件。）")

    title("第 5 步完成 ✓")
    print(f" 测试集（未见过年份）：TF 整句 {tf_t:.2%}   自回归整句 {ar_t:.2%}")
    print(" 下一步（第 6 步）实现 Beam Search，看能不能把自回归的准确率再往上推。")
    print()


if __name__ == "__main__":
    main()
