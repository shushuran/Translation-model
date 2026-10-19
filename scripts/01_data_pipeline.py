"""第 1 步自检脚本：把数据管道从里到外看一遍。

运行：
    python scripts/01_data_pipeline.py

它不训练任何模型，只是把「数据长什么样」彻底摊开给你看：
分词、词表、编码、teacher forcing 的错位、padding 与 mask、数据划分。
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

# 让脚本能直接 import 到项目根目录下的 seq2seq_attn 包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Windows 控制台默认可能是 GBK，强制 UTF-8 以免中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import numpy as np

from seq2seq_attn import data as D
from seq2seq_attn.config import CONFIG

LINE = "=" * 78
SUB = "-" * 78


def title(text: str) -> None:
    print()
    print(LINE)
    print(f" {text}")
    print(LINE)


def mask_str(mask_row) -> str:
    """把 0/1 mask 画成 ###... 的条纹，方便肉眼对齐。"""
    return "".join("#" if v > 0.5 else "." for v in mask_row)


def ids_str(vocab: D.Vocab, ids) -> str:
    return " ".join(vocab.itos[int(i)] for i in ids)


def main() -> None:
    cfg = CONFIG.data

    title("第 1 步 · 数据管道自检")
    print("任务：中文日期 → 英文日期")
    print(f"数据配置：{cfg}")

    train, dev, test = D.build_datasets(cfg)
    print(f"\n数据集大小：train={len(train)}  dev={len(dev)}  test={len(test)}")

    # ---------------------------------------------------------------- [1] 样例
    title("[1] 语料样例（train 前 12 条）")
    for ex in train.examples[:12]:
        print(f"  {ex.src_text:<24} → {ex.tgt_text}")

    # ---------------------------------------------------------------- [2] 分词
    title("[2] 分词（关键设计：数字逐位切分）")
    for ex in train.examples[:5]:
        print(f"  中文: {ex.src_text}")
        print(f"        {list(ex.src_tokens)}")
        print(f"  英文: {ex.tgt_text}")
        print(f"        {list(ex.tgt_tokens)}")
        print(SUB)

    # ---------------------------------------------------------------- [3] 词表
    title("[3] 词表（只用训练集构建）")
    print(f"源词表 size={len(train.src_vocab)}")
    print(f"  {train.src_vocab.itos}")
    print(f"\n目标词表 size={len(train.tgt_vocab)}")
    print(f"  {train.tgt_vocab.itos}")
    print(f"\n特殊符号 id： pad={train.tgt_vocab.pad_id}  bos={train.tgt_vocab.bos_id}"
          f"  eos={train.tgt_vocab.eos_id}  unk={train.tgt_vocab.unk_id}")

    # ---------------------------------------------------------------- [4] 编码
    title("[4] 编码 + teacher forcing 的「错位一位」")
    ex = train.examples[0]
    src_ids = train.src_ids[0]
    tgt_ids = train.tgt_ids[0]
    tgt_in = tgt_ids[:-1]
    tgt_out = tgt_ids[1:]
    print(f"样本: {ex.src_text}  →  {ex.tgt_text}\n")
    print(f"  src_ids  = {src_ids}")
    print(f"             {ids_str(train.src_vocab, src_ids)}")
    print()
    print(f"  tgt_ids  = {tgt_ids}")
    print(f"             {ids_str(train.tgt_vocab, tgt_ids)}   ← <bos> + 目标 + <eos>")
    print()
    print(f"  tgt_in   = {tgt_in}")
    print(f"             {ids_str(train.tgt_vocab, tgt_in)}")
    print(f"  tgt_out  = {tgt_out}")
    print(f"             {ids_str(train.tgt_vocab, tgt_out)}")
    print()
    print("  解码器第 t 步：输入 tgt_in[t]，要求输出 tgt_out[t]。")
    print("  两者错开一位 —— 这就是 teacher forcing：每一步都拿真实的上一个词当输入。")

    # ---------------------------------------------------------------- [5] Batch
    title("[5] 一个 mini-batch 的 shape 与 padding mask")
    batch = D.collate(train, list(range(0, 8)))
    print(f"  src      {str(batch.src.shape):<12} int64")
    print(f"  src_len  {str(batch.src_len.shape):<12} {batch.src_len.tolist()}")
    print(f"  src_mask {str(batch.src_mask.shape):<12} float32")
    print(f"  tgt_in   {str(batch.tgt_in.shape):<12} int64")
    print(f"  tgt_out  {str(batch.tgt_out.shape):<12} int64")
    print(f"  tgt_len  {str(batch.tgt_len.shape):<12} {batch.tgt_len.tolist()}")
    print(f"  tgt_mask {str(batch.tgt_mask.shape):<12} float32")
    print()
    print("  源句（# = 真实 token，. = <pad>）：")
    for i in range(batch.batch_size):
        raw = " ".join(train.src_vocab.itos[int(j)] for j in batch.src[i])
        print(f"    [{mask_str(batch.src_mask[i])}]  len={int(batch.src_len[i]):>2}   {raw}")
    print()
    print("  目标输入 tgt_in：")
    for i in range(batch.batch_size):
        raw = " ".join(train.tgt_vocab.itos[int(j)] for j in batch.tgt_in[i])
        print(f"    [{mask_str(batch.tgt_mask[i])}]  len={int(batch.tgt_len[i]):>2}   {raw}")
    print()
    print("  加性 attn_mask（真实位置 0，padding 位置 -1e9）：")
    am = batch.attn_mask()
    pad_row = int(np.argmin(batch.src_len))          # 挑一条有 padding 的行来看
    print(f"    shape={am.shape}")
    print(f"    第 0 行（无 padding）：{np.unique(am[0])}")
    print(f"    第 {pad_row} 行（有 padding）：{am[pad_row]}")
    print(f"    第 {pad_row} 行 mask        ：{batch.src_mask[pad_row]}")

    # ---------------------------------------------------------------- [6] 不变量
    title("[6] 自检：几条必须成立的不变量")

    # (a) tgt_in 和 tgt_out 必须错开一位
    ok_shift = np.array_equal(batch.tgt_in[:, 1:], batch.tgt_out[:, :-1])
    print(f"  (a) tgt_in[:, 1:] == tgt_out[:, :-1]  →  {ok_shift}   （teacher forcing 错位）")

    # (b) padding 位置必须是 pad_id
    ok_pad = bool(np.all(batch.src[batch.src_mask < 0.5] == train.src_vocab.pad_id))
    print(f"  (b) src 中 mask=0 的位置全为 <pad>     →  {ok_pad}")

    # (c) mask 之和必须等于长度
    ok_len = bool(np.array_equal(batch.src_mask.sum(axis=1).astype(np.int64), batch.src_len))
    print(f"  (c) src_mask.sum(1) == src_len         →  {ok_len}")

    # (d) 编码解码往返一致
    bad = 0
    for ds in (train, dev, test):
        for i in range(len(ds)):
            toks = ds.tgt_vocab.decode(ds.tgt_ids[i])
            if D.detokenize(toks) != ds.examples[i].tgt_text:
                bad += 1
    print(f"  (d) decode(tgt_ids) 还原英文原句       →  {'OK' if bad == 0 else f'{bad} 条不一致'}")

    # (e) 没有未登录词
    out_of_vocab_src = D.oov_tokens(train, train.src_vocab, "src") | D.oov_tokens(dev, train.src_vocab, "src") | D.oov_tokens(test, train.src_vocab, "src")
    out_of_vocab_tgt = D.oov_tokens(train, train.tgt_vocab, "tgt") | D.oov_tokens(dev, train.tgt_vocab, "tgt") | D.oov_tokens(test, train.tgt_vocab, "tgt")
    print(f"  (e) 未登录词：src={out_of_vocab_src or '无'}  tgt={out_of_vocab_tgt or '无'}")

    # ---------------------------------------------------------------- [7] 长度分布
    title("[7] 序列长度分布（决定计算量，也决定 padding 有多浪费）")
    for ds in (train, dev, test):
        sl = Counter(len(s) for s in ds.src_ids)
        tl = Counter(len(s) - 1 for s in ds.tgt_ids)
        print(f"  {ds.split:<6} 源长 {min(sl)}~{max(sl)} 均值 {np.mean([len(s) for s in ds.src_ids]):.1f}"
              f"   |  目标长 {min(tl)}~{max(tl)} 均值 {np.mean([len(s)-1 for s in ds.tgt_ids]):.1f}")

    # ---------------------------------------------------------------- [8] 组合泛化
    title("[8] 数据划分：测试年份在训练中从未出现（组合泛化）")
    tr_years = {e.year for e in train.examples}
    dv_years = {e.year for e in dev.examples}
    te_years = {e.year for e in test.examples}
    print(f"  train 年份 {min(tr_years)}~{max(tr_years)}（{len(tr_years)} 个）")
    print(f"  dev   年份 {min(dv_years)}~{max(dv_years)}（{len(dv_years)} 个）")
    print(f"  test  年份 {min(te_years)}~{max(te_years)}（{len(te_years)} 个）")
    print(f"  train ∩ test = {tr_years & te_years or '空集'}   → 测试年份确实没见过")
    print(f"  train ∩ dev  = {tr_years & dv_years or '空集'}")
    print("  但所有数字位 '0'~'9' 都在训练集里，所以模型只要学会「照着源句抄数字」，")
    print("  就能翻译没见过的年份 —— 这正是我们想验证的能力。")

    # ---------------------------------------------------------------- [9] 月份覆盖
    title("[9] 月份覆盖检查（12 个月必须都出现，否则模型学不会）")
    for ds in (train, dev, test):
        months = sorted({e.month for e in ds.examples})
        print(f"  {ds.split:<6} 覆盖月份 {len(months):>2}/12  {months}")
    missing = [m for m in range(1, 13) if D.MONTH_EN[m - 1] not in train.tgt_vocab.stoi]
    print(f"  训练集词表里缺失的月份名：{missing or '无'}")

    title("第 1 步完成 ✓")
    print("下一步（第 2 步）将单独实现注意力机制并用数值验证它。")
    print()


if __name__ == "__main__":
    main()
