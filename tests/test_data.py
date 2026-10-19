"""数据管道的断言测试（无第三方测试框架依赖）。

运行：
    python tests/test_data.py
"""

from __future__ import annotations

import pathlib
import random
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

PASSED = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        raise AssertionError(name)
    PASSED += 1


def test_tokenize() -> None:
    print("\n== tokenize ==")
    check("中文日期", D.tokenize("2024年3月5日") == ["2", "0", "2", "4", "年", "3", "月", "5", "日"])
    check("补零写法", D.tokenize("2024年03月05日")
          == ["2", "0", "2", "4", "年", "0", "3", "月", "0", "5", "日"])
    check("英文日期", D.tokenize("5/March/2024") == ["5", "/", "March", "/", "2", "0", "2", "4"])
    check("空串", D.tokenize("") == [])


def test_example() -> None:
    print("\n== build_example ==")
    rng = random.Random(0)
    ex = D.build_example(2024, 3, 5, rng, pad_variant_prob=0.0)
    check("源句无补零", ex.src_text == "2024年3月5日", ex.src_text)
    check("目标句", ex.tgt_text == "5/March/2024", ex.tgt_text)
    check("星期正确性", D.build_example(2024, 3, 5, rng, include_weekday=True).tgt_text
          == "5/March/2024/Tuesday")
    # 2024-03-05 是星期二
    check("闰年 2 月 29 日可生成", D.build_example(2024, 2, 29, rng).tgt_text == "29/February/2024")


def test_generate_corpus() -> None:
    print("\n== generate_corpus ==")
    rng = random.Random(0)
    corpus = D.generate_corpus(50, (2000, 2005), rng)
    check("数量正确", len(corpus) == 50)
    keys = {(e.year, e.month, e.day) for e in corpus}
    check("日期不重复", len(keys) == 50)
    check("年份都在区间内", all(2000 <= e.year <= 2005 for e in corpus))
    try:
        D.generate_corpus(10_000, (2000, 2000), rng)
        check("超额取样应报错", False)
    except ValueError:
        check("超额取样报错", True)


def test_vocab() -> None:
    print("\n== Vocab ==")
    v = D.Vocab.build([["a", "b", "a"], ["b", "c"], ["a"]])   # 频次 a=3 b=2 c=1
    check("特殊符号在最前", v.itos[:4] == ["<pad>", "<bos>", "<eos>", "<unk>"])
    check("pad_id == 0", v.pad_id == 0 and v.bos_id == 1 and v.eos_id == 2 and v.unk_id == 3)
    check("按频次从高到低排序", v.itos[4:] == ["a", "b", "c"], str(v.itos))
    check("同频次按字典序", D.Vocab.build([["b"], ["a"]]).itos[4:] == ["a", "b"])
    check("min_freq 过滤低频词", D.Vocab.build([["a"], ["b"], ["a"]], min_freq=2).itos ==
          ["<pad>", "<bos>", "<eos>", "<unk>", "a"])
    check("encode 默认不加特殊符号", v.encode(["a", "b"]) == [4, 5])
    check("encode 加 bos/eos", v.encode(["a"], True, True) == [1, 4, 2])
    check("未知词映射到 unk", v.encode(["zzz"]) == [3])
    check("decode 过滤特殊符号", v.decode([1, 4, 2, 0]) == ["a"])
    check("decode 保留特殊符号", v.decode([1, 4, 2], strip_special=False) == ["<bos>", "a", "<eos>"])
    check("json 往返", D.Vocab.from_json(v.to_json()).itos == v.itos)


def test_batch() -> None:
    print("\n== collate / make_batches ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    batch = D.collate(train, list(range(16)))

    check("src 是二维", batch.src.ndim == 2)
    check("src 宽度 = 本 batch 最大源长",
          batch.src.shape[1] == max(len(train.src_ids[i]) for i in range(16)))
    check("tgt_in/tgt_out 同形", batch.tgt_in.shape == batch.tgt_out.shape)

    # teacher forcing 错位：tgt_in[t+1] 必须等于 tgt_out[t]
    # 注意只在「t 不是本行最后一位」时成立：
    #   最后一位 tgt_out 是 <eos>，它的「下一个输入」超出了 tgt_in 的范围。
    width = batch.tgt_in.shape[1]
    pos = np.arange(width - 1)[None, :]
    valid = pos < (batch.tgt_len - 1)[:, None]
    shifted = np.array_equal(batch.tgt_in[:, 1:][valid], batch.tgt_out[:, :-1][valid])
    check("tgt_in 与 tgt_out 错开一位", shifted)

    # mask 与 pad 一致
    pad_id = train.src_vocab.pad_id
    check("mask=0 处为 pad", bool(np.all(batch.src[batch.src_mask < 0.5] == pad_id)))
    check("mask=1 处非 pad", bool(np.all(batch.src[batch.src_mask > 0.5] != pad_id)))
    check("mask.sum == len", bool(np.array_equal(batch.src_mask.sum(1).astype(np.int64), batch.src_len)))

    # 加性 mask
    am = batch.attn_mask()
    check("attn_mask 形状", am.shape == batch.src_mask.shape)
    check("attn_mask 真实位为 0", bool(np.all(am[batch.src_mask > 0.5] == 0.0)))
    check("attn_mask 填充位为负", bool(np.all(am[batch.src_mask < 0.5] < -1e8)))

    # tgt_in 的第 0 列必须是 <bos>
    check("tgt_in 首列为 <bos>", bool(np.all(batch.tgt_in[:, 0] == train.tgt_vocab.bos_id)))
    # tgt_out 的最后一个真实位置必须是 <eos>
    last = batch.tgt_out[np.arange(batch.batch_size), (batch.tgt_len - 1).astype(np.int64)]
    check("tgt_out 末位为 <eos>", bool(np.all(last == train.tgt_vocab.eos_id)))

    # drop_last
    small = train.subset(100)
    bs = D.make_batches(small, batch_size=32, shuffle=False, drop_last=True)
    check("drop_last 丢弃不足一批", len(bs) == 3 and all(b.batch_size == 32 for b in bs))
    bs2 = D.make_batches(small, batch_size=32, shuffle=False)
    check("不丢弃时最后一批更小", len(bs2) == 4 and bs2[-1].batch_size == 4)

    # shuffle 与 seed 决定顺序
    s64 = train.subset(64)
    b1 = D.make_batches(s64, 8, shuffle=True, rng=random.Random(7))[0].src[:, 0].tolist()
    b2 = D.make_batches(s64, 8, shuffle=True, rng=random.Random(7))[0].src[:, 0].tolist()
    check("相同 seed 顺序一致", b1 == b2)


def test_year_split_compositional() -> None:
    """数据划分必须真的在考「组合泛化」，而不是给模型一条背诵的捷径。

    这是第 5 步训练时用实测数据纠正出来的设计缺陷：
    最初的 train_years=(2000,2029) 只有 30 个年份，而训练样本有 4000 条，
    等于每个年份平均出现 133 次 —— 模型直接背下了这 30 个年份：
    训练集整句准确率 97.8%，测试集 0%，而且生成的年份**从不超出**训练年份集合。
    同时 2000~2029 的年份第三位只能是 0/1/2，测试年份的第三位是 3，
    这个「位置上的数字」训练时根本没见过。
    """
    print("\n== 年份划分的组合泛化性质 ==")
    train, dev, test = D.build_datasets(CONFIG.data)

    # ---- 要求 1：每个数位上出现过的数字，训练集都要覆盖 ----
    def position_digits(examples):
        sets = [set() for _ in range(4)]
        for e in examples:
            s = str(e.year)
            for i, ch in enumerate(s):
                sets[len(s) - 1 - i].add(ch)
        return sets

    def ffill(sets):
        """把右侧对齐的位补齐到 4 位（不同年份位数可能不同）。"""
        out = [set() for _ in range(4)]
        for i, s in enumerate(sets):
            out[4 - len(sets) + i] = s
        return out

    train_pos = ffill(position_digits(train.examples))
    check("训练集 4 个数位都有非空数字集合", all(len(s) > 0 for s in train_pos),
          str([sorted(s) for s in train_pos]))
    for name, ds in (("dev", dev), ("test", test)):
        ds_pos = ffill(position_digits(ds.examples))
        missing = []
        for pos in range(4):
            for ch in ds_pos[pos]:
                if ch not in train_pos[pos]:
                    missing.append((pos, ch))
        check(f"{name} 集每个数位的数字都在训练集出现过", not missing, str(missing))

    # ---- 要求 2：「样本数 / 不同年份数」接近 1，背诵不可能奏效 ----
    years = [e.year for e in train.examples]
    distinct = len(set(years))
    ratio = len(years) / distinct
    check("训练集每个年份平均出现次数 < 2（背诵不可行）", ratio < 2.0,
          f"{len(years)} 条 / {distinct} 个年份 = {ratio:.2f} 次/年")
    check("训练年份数量足够多", distinct > 1000, str(distinct))

    # ---- 要求 3：测试年份与训练年份完全不重叠 ----
    tr, dv, te = set(years), {e.year for e in dev.examples}, {e.year for e in test.examples}
    check("train ∩ test = 空集", not (tr & te))
    check("train ∩ dev = 空集", not (tr & dv))
    check("dev ∩ test = 空集", not (dv & te))

    # ---- 要求 4：测试年份确实不在训练年份集合里（逐条断言）----
    leaked = [e.year for e in test.examples if e.year in tr]
    check("没有任何测试年份泄露进训练集", not leaked, str(leaked[:5]))


def test_roundtrip_and_split() -> None:
    print("\n== 往返一致 与 数据划分 ==")
    train, dev, test = D.build_datasets(CONFIG.data)
    bad = 0
    for ds in (train, dev, test):
        for i in range(len(ds)):
            if D.detokenize(ds.tgt_vocab.decode(ds.tgt_ids[i])) != ds.examples[i].tgt_text:
                bad += 1
            if D.detokenize(ds.src_vocab.decode(ds.src_ids[i])) != ds.examples[i].src_text:
                bad += 1
    check("所有样本 en/decode 往返一致", bad == 0, f"不一致 {bad} 条")

    check("无源端未登录词",
          not (D.oov_tokens(dev, train.src_vocab, "src") | D.oov_tokens(test, train.src_vocab, "src")))
    check("无目标端未登录词",
          not (D.oov_tokens(dev, train.tgt_vocab, "tgt") | D.oov_tokens(test, train.tgt_vocab, "tgt")))

    tr = {e.year for e in train.examples}
    te = {e.year for e in test.examples}
    dv = {e.year for e in dev.examples}
    check("测试年份与训练不重叠", not (tr & te))
    check("开发年份与训练不重叠", not (tr & dv))
    check("月份全覆盖 12 个月",
          all(m in train.tgt_vocab.stoi for m in D.MONTH_EN))

    # 数据可复现
    t2, _, _ = D.build_datasets(CONFIG.data)
    check("相同 seed 数据完全一致",
          [e.src_text for e in train.examples[:50]] == [e.src_text for e in t2.examples[:50]])


def main() -> None:
    print("=" * 70)
    print(" 数据管道测试")
    print("=" * 70)
    test_tokenize()
    test_example()
    test_generate_corpus()
    test_vocab()
    test_batch()
    test_year_split_compositional()
    test_roundtrip_and_split()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
