"""第 8 步：消融实验 —— 量化「注意力到底带来了多少」。

用法：
    python scripts/08_ablation.py --list                       # 列出所有配置
    python scripts/08_ablation.py --all                        # 全部跑一遍（慢，约 15 分钟）
    python scripts/08_ablation.py --run no_attention --n-train 500   # 跑单个配置
    python scripts/08_ablation.py --report                     # 汇总已有结果

每个配置单独跑一次训练（不共享任何东西），结果存成
`ablation/<配置名>_n<训练条数>.json`，所以可以分多个进程并行跑，
再随时用 `--report` 汇总。

为什么要做「数据受限 + 完整数据」两套
-----------------------------------
完整数据（4000 条）下全模型 700 步就 100% 了，很多消融项也会是 100%，
表格里全是 1.0000，消融"消"不出信息。
把训练集压到 500 条后任务进入**数据受限**区间，各配置的差距才会显现。
两套结果并排看，还能回答"注意力在数据充足时是否仍然关键"。
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
from seq2seq_attn import infer as I
from seq2seq_attn import train as T
from seq2seq_attn import viz as V
from seq2seq_attn.config import CONFIG

LINE = "=" * 82
SUB = "-" * 82
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ablation"

# 配置名 -> ModelConfig 的覆盖项。每一项都是**单变量**修改。
ABLATIONS: dict[str, dict] = {
    "baseline":     {},                                        # 完整模型（对照）
    "no_attention": {"attn_type": "mean"},                     # 去掉注意力：固定 context 向量
    "dot":          {"attn_type": "dot"},                      # 换缩放点积（零参数）
    "output_s":     {"decoder_output": "s"},                   # 输出层只吃 s_t
    "output_s_emb": {"decoder_output": "s_c_emb"},             # 输出层还吃 emb(y_{t-1})
    "init_zero":    {"decoder_init": "zero"},                  # 解码器起点恒为零
}

DESCRIPTIONS = {
    "baseline": "完整模型（加性注意力 + [s;c] + bridge）",
    "no_attention": "去掉注意力：context = 编码器输出的掩码平均（固定向量）",
    "dot": "缩放点积注意力（零参数，对比加性的 33k 参数）",
    "output_s": "输出层只吃 s_t（context 只能经 GRU 输入影响输出）",
    "output_s_emb": "输出层吃 [s_t; c_t; emb(y_{t-1})]（Bahdanau 原论文）",
    "init_zero": "解码器初始状态恒为零（全靠注意力提供源句信息）",
}


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def make_config(name: str, n_train: int):
    overrides = ABLATIONS[name]
    data_cfg = dataclasses.replace(CONFIG.data, n_train=n_train)
    model_cfg = dataclasses.replace(CONFIG.model, **overrides)
    return dataclasses.replace(CONFIG, data=data_cfg, model=model_cfg)


def result_path(name: str, n_train: int) -> pathlib.Path:
    return OUT_DIR / f"{name}_n{n_train}.json"


# ======================================================================================
def run_one(name: str, n_train: int, steps: int) -> dict:
    cfg = make_config(name, n_train)
    print(f"  [{name} | n_train={n_train}] 开始训练 {steps} 步 ……", flush=True)

    t0 = time.perf_counter()
    res = T.train(cfg, max_steps=steps, verbose=False, eval_greedy=True,
                  eval_every=max(50, steps // 10))
    elapsed = time.perf_counter() - t0

    model = res["model"]
    dev_set, test_set, max_len = res["dev_set"], res["test_set"], res["max_len"]

    # 收敛速度：第一次在开发集上达到 99% 整句准确率的步数
    hist = res["history"]
    steps_to_99 = None
    for s, acc in zip(hist["eval_step"], hist["dev_tf_seq"]):
        if acc >= 0.99:
            steps_to_99 = int(s) + 1
            break

    tf_dev = T.evaluate_teacher_forcing(model, dev_set)
    ar_dev = I.evaluate_decoding(model, dev_set, max_len, method="greedy")
    tf_test = T.evaluate_teacher_forcing(model, test_set)
    ar_test = I.evaluate_decoding(model, test_set, max_len, method="greedy")
    beam_test = I.evaluate_decoding(model, test_set, max_len, method="beam", beam_size=3)

    # 对齐质量（"mean" 模式下权重恒为均匀分布，算出来就是随机水平，仅作参考）
    rows = V.build_alignments(model, test_set, range(min(200, len(test_set))))
    align = V.alignment_stats(rows)

    record = {
        "name": name,
        "description": DESCRIPTIONS.get(name, ""),
        "n_train": n_train,
        "steps": steps,
        "elapsed": elapsed,
        "params_total": model.num_params(),
        "params_attn": model.dec.attn.num_params(),
        "train_loss_final": float(np.mean(hist["loss"][-50:])),
        "steps_to_99": steps_to_99,
        "dev_tf_token": tf_dev["token_acc"],
        "dev_tf_seq": tf_dev["seq_acc"],
        "dev_ar_token": ar_dev["token_acc"],
        "dev_ar_seq": ar_dev["seq_acc"],
        "test_tf_token": tf_test["token_acc"],
        "test_tf_seq": tf_test["seq_acc"],
        "test_ar_token": ar_test["token_acc"],
        "test_ar_seq": ar_test["seq_acc"],
        "test_beam3_token": beam_test["token_acc"],
        "test_beam3_seq": beam_test["seq_acc"],
        "align_argmax_hit": align["argmax_hit"],
        "align_year_exact": align["year_position_exact"],
        "align_cross_decreasing": align["cross_field_decreasing"],
    }

    OUT_DIR.mkdir(exist_ok=True)
    result_path(name, n_train).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  [{name} | n_train={n_train}] 完成  用时 {elapsed:.1f}s  "
          f"参数 {record['params_total']:,}（注意力 {record['params_attn']:,}）  "
          f"test AR 整句 {record['test_ar_seq']:.4f}", flush=True)
    return record


# ======================================================================================
def collect(n_train: int | None = None) -> list[dict]:
    if not OUT_DIR.exists():
        return []
    out = []
    for p in sorted(OUT_DIR.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if n_train is None or rec["n_train"] == n_train:
            out.append(rec)
    return out


def report() -> None:
    records = collect()
    if not records:
        print("  还没有任何结果，先运行 --run 或 --all")
        return

    n_list = sorted({r["n_train"] for r in records})
    for n_train in n_list:
        group = [r for r in records if r["n_train"] == n_train]
        order = {name: i for i, name in enumerate(ABLATIONS)}
        group.sort(key=lambda r: order.get(r["name"], 99))

        title(f"消融结果：训练集 {n_train} 条（测试集 400 条，年份训练时从未出现）")
        print(f"  {'配置':<14}{'参数量':>9}{'注意力参数':>11}{'收敛步数':>10}"
              f"{'dev TF整句':>11}{'test TF整句':>12}{'test AR整句':>12}{'对齐命中':>10}")
        print("  " + "-" * 92)
        base = next((r for r in group if r["name"] == "baseline"), None)
        for r in group:
            s99 = "-" if r["steps_to_99"] is None else str(r["steps_to_99"])
            print(f"  {r['name']:<14}{r['params_total']:>9,}{r['params_attn']:>11,}"
                  f"{s99:>10}{r['dev_tf_seq']:>11.4f}{r['test_tf_seq']:>12.4f}"
                  f"{r['test_ar_seq']:>12.4f}{r['align_argmax_hit']:>10.4f}")

        if base is not None:
            print(f"\n  相对 baseline 的 test AR 整句准确率变化：")
            for r in group:
                if r["name"] == "baseline":
                    continue
                d = r["test_ar_seq"] - base["test_ar_seq"]
                mark = "  ← 明显下降" if d < -0.02 else ("  ← 有提升" if d > 0.02 else "")
                print(f"    {r['name']:<16}{d:>+8.4f}{mark}")

        print(f"\n  配置说明：")
        for r in group:
            print(f"    {r['name']:<16}{r['description']}")

    # ---- 跨条件对比：数据量放大还是缩小了各配置的差距 ----
    if len(n_list) > 1:
        title("数据量的影响：同一配置在不同训练集大小下的 test AR 整句准确率")
        names = [n for n in ABLATIONS if any(r["name"] == n for r in records)]
        print(f"  {'配置':<14}" + "".join(f"{'n=' + str(n):>12}" for n in n_list)
              + f"{'最大提升':>12}")
        print("  " + "-" * (14 + 12 * len(n_list) + 12))
        for name in names:
            row = [next((r for r in records
                         if r["name"] == name and r["n_train"] == n), None) for n in n_list]
            if any(r is None for r in row):
                continue
            vals = [r["test_ar_seq"] for r in row]
            gain = max(vals) - min(vals)
            print(f"  {name:<14}" + "".join(f"{v:>12.4f}" for v in vals)
                  + f"{gain:>+12.4f}")
        print("\n  读法：把训练集从 500 加到 4000 后，如果某个配置提升很小，")
        print("  说明它受限于**表达/结构**而不是数据量；如果提升很大，")
        print("  说明它的瓶颈原本是数据不足。")


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="第 8 步：消融实验")
    ap.add_argument("--list", action="store_true", help="列出所有配置")
    ap.add_argument("--run", type=str, default=None, help="跑单个配置")
    ap.add_argument("--n-train", type=int, default=500, help="训练集大小（默认 500）")
    ap.add_argument("--steps", type=int, default=1500, help="每个配置训练步数")
    ap.add_argument("--report", action="store_true", help="只汇总已有结果")
    ap.add_argument("--all", action="store_true", help="跑全部配置（两套数据设置）")
    ap.add_argument("--full-only", action="store_true", help="--all 时只跑完整数据那一套")
    args = ap.parse_args()

    print(LINE)
    print(" 第 8 步 · 消融实验")
    print(LINE)

    if args.list:
        print(f"  {'配置名':<16}{'修改项':<44}说明")
        print("  " + "-" * 100)
        for name, ov in ABLATIONS.items():
            print(f"  {name:<16}{str(ov):<44}{DESCRIPTIONS[name]}")
        return

    if args.report:
        report()
        return

    if args.run:
        if args.run not in ABLATIONS:
            print(f"  未知配置 {args.run!r}，可用：{list(ABLATIONS)}")
            raise SystemExit(1)
        run_one(args.run, args.n_train, args.steps)
        print()
        report()
        return

    if args.all:
        settings = [4000] if args.full_only else [500, 4000]
        for n_train in settings:
            for name in ABLATIONS:
                run_one(name, n_train, args.steps)
        print()
        report()
        return

    ap.print_help()


if __name__ == "__main__":
    main()
