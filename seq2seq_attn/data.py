"""数据管道：日期语料生成 → 分词 → 词表 → 编码 → padding → batch。

任务：中文日期 → 英文日期
    源句（中文）  「2024年3月5日」
    目标句（英文）「5/March/2024」

分词规则（这是本练习的第一个关键设计）：
    数字串逐位切分，连续的英文字母整体成一个词，其余非空白字符各自成词。
        「2024年3月5日」  →  ['2','0','2','4','年','3','月','5','日']   (9 个 token)
        「2024年03月05日」→  ['2','0','2','4','年','0','3','月','0','5','日']
        「5/March/2024」  →  ['5','/','March','/','2','0','2','4']

    为什么不把英文也拆成字母？因为月份名一共只有 12 个，是**封闭集合**，
    让 'March' 当一个整体词，模型只需要学「中文的月数字 → 英文月份名」这一条映射；
    拆成字母反而把一次查表变成五次字符生成，没有额外收益。

这样做的好处：
    1. 词表极小（约 20 个符号），embedding 学得动，训练几分钟就能收敛；
    2. 年份的「数字位」在训练/测试集之间共享，所以可以用**没见过的年份**做测试，
       考查的是真的学会了「复制数字」，而不是把 200 个年份词背下来。
"""

from __future__ import annotations

import calendar
import datetime as _dt
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# 特殊符号
# --------------------------------------------------------------------------------------
PAD = "<pad>"   # 填充，id 固定为 0
BOS = "<bos>"   # 序列起始
EOS = "<eos>"   # 序列结束
UNK = "<unk>"   # 未登录词

SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS, UNK)

MONTH_EN: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

WEEKDAY_ZH: tuple[str, ...] = ("一", "二", "三", "四", "五", "六", "日")
WEEKDAY_EN: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)

# 匹配「一段连续数字」/「一段连续英文字母」/「单个其它非空白字符」
_TOKEN_RE = re.compile(r"\d+|[A-Za-z]+|[^\d\sA-Za-z]")

# 加性 mask 用的极小值。用 -1e9 而不是 -inf，
# 是为了避免 0 * inf = nan（反向传播时容易踩坑）。
NEG_INF = -1e9


# --------------------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """把字符串切成 token 列表：数字逐位，英文字母成词，其它字符逐个。"""
    tokens: list[str] = []
    for chunk in _TOKEN_RE.findall(text):
        if chunk.isdigit():
            tokens.extend(chunk)          # "2024"  -> ['2','0','2','4']
        else:
            tokens.append(chunk)          # "年"/"March" -> ['年']/['March']
    return tokens


def detokenize(tokens: Iterable[str]) -> str:
    """token 列表拼回字符串（仅用于展示；英文需要分隔符还原）。"""
    return "".join(tokens)


# --------------------------------------------------------------------------------------
# 语料生成
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Example:
    """一条平行语料样本。"""

    src_text: str
    tgt_text: str
    src_tokens: tuple[str, ...]
    tgt_tokens: tuple[str, ...]
    year: int
    month: int
    day: int

    def __str__(self) -> str:
        return f"{self.src_text}  →  {self.tgt_text}"


def build_example(
    year: int,
    month: int,
    day: int,
    rng: random.Random,
    pad_variant_prob: float = 0.5,
    include_weekday: bool = False,
) -> Example:
    """用指定的年月日造一条样本。"""
    weekday = _dt.date(year, month, day).weekday()   # 0=周一

    # ---- 中文源句 ----
    if rng.random() < pad_variant_prob:
        zh = f"{year}年{month:02d}月{day:02d}日"     # 2024年03月05日
    else:
        zh = f"{year}年{month}月{day}日"             # 2024年3月5日
    if include_weekday:
        zh += f"星期{WEEKDAY_ZH[weekday]}"

    # ---- 英文目标句（固定规范格式，不带前导零）----
    en = f"{day}/{MONTH_EN[month - 1]}/{year}"
    if include_weekday:
        en += f"/{WEEKDAY_EN[weekday]}"

    return Example(
        src_text=zh,
        tgt_text=en,
        src_tokens=tuple(tokenize(zh)),
        tgt_tokens=tuple(tokenize(en)),
        year=year,
        month=month,
        day=day,
    )


def generate_corpus(
    n: int,
    year_range: tuple[int, int],
    rng: random.Random,
    pad_variant_prob: float = 0.5,
    include_weekday: bool = False,
) -> list[Example]:
    """在给定年份区间内随机抽 n 个**互不重复**的日期，生成平行语料。

    去重很重要：如果同一日期重复出现，模型会靠记忆而不是靠学习。
    """
    y_lo, y_hi = year_range
    pool = [
        (y, m, d)
        for y in range(y_lo, y_hi + 1)
        for m in range(1, 13)
        for d in range(1, calendar.monthrange(y, m)[1] + 1)
    ]
    if n > len(pool):
        raise ValueError(
            f"年份区间 {year_range} 只有 {len(pool)} 个不同日期，无法抽取 {n} 条不重复样本"
        )
    rng.shuffle(pool)
    return [
        build_example(y, m, d, rng, pad_variant_prob, include_weekday)
        for (y, m, d) in pool[:n]
    ]


# --------------------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------------------
class Vocab:
    """极简词表：id 0..3 固定留给 <pad> <bos> <eos> <unk>。"""

    def __init__(self, itos: Sequence[str]) -> None:
        self.itos: list[str] = list(itos)
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("词表里有重复 token")
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    # ---- 构造 ----
    @classmethod
    def build(cls, token_seqs: Iterable[Sequence[str]], min_freq: int = 1) -> "Vocab":
        """按频次从高到低排序建表；频次相同时按字典序，保证可复现。"""
        counter = Counter(t for seq in token_seqs for t in seq)
        items = [t for t, c in counter.items() if c >= min_freq]
        items.sort(key=lambda t: (-counter[t], t))
        return cls(list(SPECIAL_TOKENS) + items)

    # ---- 特殊 id ----
    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)}, tokens={self.itos})"

    # ---- 编解码 ----
    def encode(self, tokens: Sequence[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> list[str]:
        out: list[str] = []
        for i in ids:
            token = self.itos[int(i)]
            if strip_special and token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return out

    # ---- 存取 ----
    def to_json(self) -> str:
        return json.dumps({"itos": self.itos}, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Vocab":
        return cls(json.loads(text)["itos"])


# --------------------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------------------
@dataclass
class DateDataset:
    """把 Example 列表 + 词表打包成可按索引取 id 序列的数据集。"""

    examples: list[Example]
    src_vocab: Vocab
    tgt_vocab: Vocab
    split: str = "train"

    # 以下字段在 __post_init__ 里填充
    src_ids: list[list[int]] = field(init=False, default_factory=list)
    tgt_ids: list[list[int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.src_ids = [self.src_vocab.encode(e.src_tokens) for e in self.examples]
        # 目标序列带 <bos> / <eos>：
        #   ids      = [<bos>] + y + [<eos>]
        #   tgt_in   = ids[:-1] = [<bos>] + y          ← 解码器第 t 步的输入
        #   tgt_out  = ids[1:]  = y + [<eos>]          ← 解码器第 t 步的监督标签
        # 二者长度相同、错开一位，这就是 teacher forcing 的实现方式。
        self.tgt_ids = [
            self.tgt_vocab.encode(e.tgt_tokens, add_bos=True, add_eos=True)
            for e in self.examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    def subset(self, n_or_indices) -> "DateDataset":
        """取子集：传 int 取前 n 条，传索引序列按索引取。

        注意会复用同一个词表对象，所以子集不会重造词表。
        """
        if isinstance(n_or_indices, int):
            examples = self.examples[:n_or_indices]
        else:
            examples = [self.examples[i] for i in n_or_indices]
        return DateDataset(examples, self.src_vocab, self.tgt_vocab, self.split)

    @property
    def max_src_len(self) -> int:
        return max(len(s) for s in self.src_ids)

    @property
    def max_tgt_len(self) -> int:
        """不含 <eos> 的目标长度（即解码步数）。"""
        return max(len(s) - 1 for s in self.tgt_ids)


# --------------------------------------------------------------------------------------
# Batch 组装
# --------------------------------------------------------------------------------------
@dataclass
class Batch:
    """一个 mini-batch 的全部张量。所有数组第 0 维都是 batch。"""

    src: np.ndarray        # (B, S) int64   编码器输入（右补 <pad>）
    src_len: np.ndarray    # (B,)    int64   每条源句的真实长度
    src_mask: np.ndarray   # (B, S)  f32     1.0=真实 token，0.0=<pad>
    tgt_in: np.ndarray     # (B, T) int64   解码器输入（<bos> + y）
    tgt_out: np.ndarray    # (B, T) int64   监督标签（y + <eos>）
    tgt_len: np.ndarray    # (B,)    int64   每条目标句的真实长度
    tgt_mask: np.ndarray   # (B, T)  f32     1.0=真实 token，0.0=<pad>
    src_texts: list[str]   # 便于调试时肉眼检查
    tgt_texts: list[str]

    @property
    def batch_size(self) -> int:
        return int(self.src.shape[0])

    def attn_mask(self, neg: float = NEG_INF) -> np.ndarray:
        """转成加性 mask：(B, S)，padding 位置为 -1e9，真实位置为 0。

        注意力里直接 `scores + attn_mask[:, None, :]`，
        softmax 之后这些位置的权重就是 0（数值上是 e^{-1e9}≈0）。
        """
        return (1.0 - self.src_mask) * neg


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """右填充到本 batch 内的最大长度，同时返回真实长度和 mask。"""
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)
    lens = np.zeros((len(seqs),), dtype=np.int64)
    for row, seq in enumerate(seqs):
        out[row, : len(seq)] = seq
        mask[row, : len(seq)] = 1.0
        lens[row] = len(seq)
    return out, lens, mask


def collate(dataset: DateDataset, indices: Sequence[int]) -> Batch:
    """把若干条样本拼成一个 Batch。"""
    src_seqs, tgt_in_seqs, tgt_out_seqs = [], [], []
    for i in indices:
        src_seqs.append(dataset.src_ids[i])
        full = dataset.tgt_ids[i]
        tgt_in_seqs.append(full[:-1])
        tgt_out_seqs.append(full[1:])

    src, src_len, src_mask = _pad(src_seqs, dataset.src_vocab.pad_id)
    tgt_in, tgt_len, tgt_mask = _pad(tgt_in_seqs, dataset.tgt_vocab.pad_id)
    tgt_out, _, _ = _pad(tgt_out_seqs, dataset.tgt_vocab.pad_id)

    return Batch(
        src=src,
        src_len=src_len,
        src_mask=src_mask,
        tgt_in=tgt_in,
        tgt_out=tgt_out,
        tgt_len=tgt_len,
        tgt_mask=tgt_mask,
        src_texts=[dataset.examples[i].src_text for i in indices],
        tgt_texts=[dataset.examples[i].tgt_text for i in indices],
    )


def make_batches(
    dataset: DateDataset,
    batch_size: int,
    shuffle: bool = True,
    rng: random.Random | None = None,
    drop_last: bool = False,
) -> list[Batch]:
    """把数据集切成 batch 列表。"""
    order = list(range(len(dataset)))
    if shuffle:
        (rng or random).shuffle(order)
    if drop_last:
        order = order[: (len(order) // batch_size) * batch_size]

    batches: list[Batch] = []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        if len(idx) < batch_size and drop_last:
            continue
        batches.append(collate(dataset, idx))
    return batches


# --------------------------------------------------------------------------------------
# 顶层入口
# --------------------------------------------------------------------------------------
def oov_tokens(dataset: DateDataset, vocab: Vocab, side: str = "src") -> set[str]:
    """返回该数据集里会变成 <unk> 的 token 集合（应为空）。"""
    bad: set[str] = set()
    for e in dataset.examples:
        tokens = e.src_tokens if side == "src" else e.tgt_tokens
        bad.update(t for t in tokens if t not in vocab.stoi)
    return bad


def build_datasets(cfg) -> tuple[DateDataset, DateDataset, DateDataset]:
    """按 DataConfig 构建 train / dev / test 三个数据集。

    词表只从**训练集**构建，这是评估时保持诚实的关键。
    """
    rng = random.Random(cfg.seed)
    kw = dict(pad_variant_prob=cfg.pad_variant_prob, include_weekday=cfg.include_weekday)

    train_ex = generate_corpus(cfg.n_train, cfg.train_years, rng, **kw)
    dev_ex = generate_corpus(cfg.n_dev, cfg.dev_years, rng, **kw)
    test_ex = generate_corpus(cfg.n_test, cfg.test_years, rng, **kw)

    src_vocab = Vocab.build(e.src_tokens for e in train_ex)
    tgt_vocab = Vocab.build(e.tgt_tokens for e in train_ex)

    return (
        DateDataset(train_ex, src_vocab, tgt_vocab, "train"),
        DateDataset(dev_ex, src_vocab, tgt_vocab, "dev"),
        DateDataset(test_ex, src_vocab, tgt_vocab, "test"),
    )
