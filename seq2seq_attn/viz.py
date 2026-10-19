"""注意力对齐可视化与定量分析。

本任务的输出格式固定为 `日/月名/年`，源句格式固定为 `[年]年[月]月[日]日`，
所以**理想对齐是可以写出来的**：

    输出「日的数字」  -> 应关注源句里「日」之前的那几位数字
    输出「/」         -> 不需要看源句
    输出「月份名」    -> 应关注源句里「年」和「月」之间的那几位数字 + '月'
    输出「年份数字」  -> 应关注源句里「年」之前的四位数字

有了理想对齐，就能把"热力图看起来挺漂亮"变成**可测量的数字**：
    · argmax 命中率：每个输出 token 注意力最大的那个源位置，是否落在理想集合里
    · 位置精确率：输出年份的第 k 位，是否正好看源句年份的第 k 位（严格 1-1 对齐）
    · 单调性：argmax 序列是否单调变化（本任务的源序和目标序是**相反**的：
      输出从"日"开始、源句从"年"开始，所以理想情况是单调**递减**）

注意：注意力权重只是"模型看了哪里"的证据，不等于"模型的决策依据"。
注意力可解释性是一个有争议的话题 —— 但要判断"模型有没有学会对齐"，
它仍然是最直接的观察窗口。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import data as D
from .model import Seq2Seq

__all__ = [
    "Alignment",
    "build_alignment",
    "build_alignments",
    "alignment_stats",
    "span_matrix",
    "canonical_average_weights",
    "render_ascii",
    "display_width",
    "pad_display",
    "save_png",
    "CANONICAL_TARGET_LABELS",
    "CANONICAL_SOURCE_LABELS",
]

# 目标 token 的类别
KIND_DAY = "day"        # 日的数字
KIND_SEP = "sep"        # 分隔符 '/'
KIND_MONTH = "month"    # 月份名
KIND_YEAR = "year"      # 年份数字
KIND_EOS = "eos"        # 序列结束符 <eos>

KIND_CN = {KIND_DAY: "日", KIND_MONTH: "月", KIND_YEAR: "年",
           KIND_SEP: "／", KIND_EOS: "止"}

# 规范化的行/列标签：把变长的源句/目标句对齐到固定格子里，
# 「右对齐」是因为中文的补零写法（03月05日）在目标里会被去掉前导零。
CANONICAL_TARGET_LABELS = ["日①", "日②", "/", "月份", "/", "年①", "年②", "年③", "年④"]
CANONICAL_SOURCE_LABELS = ["年①", "年②", "年③", "年④", "年", "月①", "月②", "月",
                           "日①", "日②", "日"]

CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


# --------------------------------------------------------------------------------------
# 文本显示宽度（CJK 占两列）
# --------------------------------------------------------------------------------------
def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad_display(text: str, width: int, align: str = "left") -> str:
    pad = " " * max(0, width - display_width(text))
    if align == "right":
        return pad + text
    if align == "center":
        half = len(pad) // 2
        return " " * half + text + " " * (len(pad) - half)
    return text + pad


# --------------------------------------------------------------------------------------
# 对齐解析
# --------------------------------------------------------------------------------------
@dataclass
class Alignment:
    """一条样本的注意力权重 + 理想对齐标注。"""

    index: int
    src_text: str
    tgt_text: str
    src_tokens: list[str]
    tgt_tokens: list[str]
    weights: np.ndarray                    # (T, S)
    year_src: list[int] = field(default_factory=list)
    month_src: list[int] = field(default_factory=list)
    day_src: list[int] = field(default_factory=list)
    month_marker: int = -1                 # '月' 这个定界符的位置
    tgt_kind: list[str] = field(default_factory=list)
    ideal: list[set] = field(default_factory=list)   # 每个目标位置的理想源位置集合
    src_len: int = 0

    def __post_init__(self) -> None:
        if self.weights.ndim != 2:
            raise ValueError(f"weights 必须是二维 (T,S)，收到 {self.weights.shape}")
        if self.weights.shape != (len(self.tgt_tokens), len(self.src_tokens)):
            raise ValueError(
                f"weights 形状 {self.weights.shape} 与 token 数 "
                f"(T={len(self.tgt_tokens)}, S={len(self.src_tokens)}) 不符"
            )
        self.src_len = len(self.src_tokens)

    # ---- 便捷访问 ----
    def positions_of_kind(self, kind: str) -> list[int]:
        return [i for i, k in enumerate(self.tgt_kind) if k == kind]

    def span_of_kind(self, kind: str) -> list[int]:
        """某一类的**数字**在源句里的位置（不含定界符）。"""
        return {"year": self.year_src, "month": self.month_src, "day": self.day_src}[kind]

    def field_of_kind(self, kind: str) -> list[int]:
        """某一类的**完整字段**在源句里的位置（含定界符）。

        与 `month_field()` 保持同一口径 —— 月份字段 = 月数字 + '月'。
        两者定义不一致的话，`span_matrix` 会在"完美对齐"上算出全零矩阵。
        """
        if kind == KIND_MONTH:
            s = set(self.month_src)
            if self.month_marker >= 0:
                s.add(self.month_marker)
            return sorted(s)
        return list(self.span_of_kind(kind))

    def month_field(self) -> set:
        """月份字段 = 月的数字 + '月' 这个定界符。

        实测发现这一步很重要：对「**10月**」这种数字有歧义的月份，
        模型会把 95% 的权重放在 `月` 上而不是 `1`/`0` 上 ——
        因为 `1`、`0` 也可能是日期或年份的一部分，定界符才是可靠的线索。
        如果理想集合只含数字，这个**正确**的行为会被误判成失败。
        """
        s = set(self.month_src)
        if self.month_marker >= 0:
            s.add(self.month_marker)
        return s

    def argmax_positions(self) -> list[int]:
        return [int(np.argmax(self.weights[t])) for t in range(self.weights.shape[0])]

    def row_entropy(self) -> np.ndarray:
        """每一行（每个输出 token）的归一化熵。"""
        p = np.clip(self.weights, 1e-12, 1.0)
        ent = -(p * np.log(p)).sum(axis=-1)
        n_valid = float((self.weights.sum(axis=-1) > 0).sum())
        norm = np.log(max(self.weights.shape[1], 2))
        return ent / norm


def _split_target(tgt_tokens: list[str]) -> tuple[list[int], int, int, int]:
    """把目标 token 切成 [日数字] '/' [月份] '/' [年数字]，返回各段的长度。

    目标格式固定为 `日/月份名/年`，所以直接按 '/' 分三段。
    """
    seps = [i for i, t in enumerate(tgt_tokens) if t == "/"]
    if len(seps) != 2:
        raise ValueError(f"目标句里应该有恰好 2 个 '/'，收到 {len(seps)}：{tgt_tokens}")
    n_day = seps[0]
    n_month = seps[1] - seps[0] - 1
    n_year = len(tgt_tokens) - seps[1] - 1
    if n_month != 1:
        raise ValueError(f"月份应该占 1 个 token，收到 {n_month}：{tgt_tokens}")
    return n_day, n_month, n_year


def _parse_source(src_tokens: list[str]) -> tuple[list[int], list[int], list[int]]:
    """按「年 / 月 / 日」三个汉字切分源 token，返回三段的索引列表。"""
    def find(ch: str) -> int:
        try:
            return src_tokens.index(ch)
        except ValueError:
            raise ValueError(f"源句里找不到 {ch!r}：{src_tokens}") from None

    i_nian, i_yue, i_ri = find("年"), find("月"), find("日")
    if not (i_nian < i_yue < i_ri):
        raise ValueError(f"源句里的 年/月/日 顺序不对：{src_tokens}")
    year = list(range(0, i_nian))
    month = list(range(i_nian + 1, i_yue))
    day = list(range(i_yue + 1, i_ri))
    if not (year and month and day):
        raise ValueError(f"源句的年/月/日 段不能为空：{src_tokens}")
    return year, month, day


def build_alignment(model: Seq2Seq, dataset: D.DateDataset, index: int,
                    max_len: int | None = None) -> Alignment:
    """取一条样本的注意力权重，并标注理想对齐。"""
    ex = dataset.examples[index]
    batch = D.collate(dataset, [index])
    h_seq, h_final = model.enc.forward(batch.src, batch.src_mask)
    _, weights = model.dec.forward(batch.tgt_in, h_seq, batch.src_mask, h_final)

    src_tokens = list(ex.src_tokens)
    # 只取真实位置（去掉 padding）—— collate 会补 <pad> 到 batch 最大长度
    real_src = int(batch.src_len[0])
    src_tokens = src_tokens[:real_src]
    w = weights[0][: int(batch.tgt_len[0]), :real_src]

    # 解码步数比可见目标多 1：最后一步生成的是 <eos>。
    # 这一步的注意力也很有意义（"模型决定停止时在看哪里"），所以保留它，
    # 但把它单列成 KIND_EOS，不计入对齐命中率。
    tgt_tokens = list(ex.tgt_tokens)
    year_src, month_src, day_src = _parse_source(src_tokens)
    month_marker = src_tokens.index("月")
    n_day, _, _ = _split_target(tgt_tokens)

    kind: list[str] = []
    ideal: list[set] = []
    for i, tok in enumerate(tgt_tokens):
        if i < n_day:
            kind.append(KIND_DAY)
            ideal.append(set(day_src))
        elif tok == "/":
            kind.append(KIND_SEP)
            ideal.append(set())
        elif i == n_day + 1:
            kind.append(KIND_MONTH)
            ideal.append(set(month_src) | {month_marker})
        else:
            kind.append(KIND_YEAR)
            ideal.append(set(year_src))

    n_steps = w.shape[0]
    if n_steps != len(tgt_tokens) + 1:
        raise ValueError(
            f"解码步数 {n_steps} 与「目标 token 数 {len(tgt_tokens)} + 1（<eos>）」不符"
        )
    tgt_tokens = tgt_tokens + ["<eos>"]
    kind.append(KIND_EOS)
    ideal.append(set())

    return Alignment(
        index=index, src_text=ex.src_text, tgt_text=ex.tgt_text,
        src_tokens=src_tokens, tgt_tokens=tgt_tokens, weights=w,
        year_src=year_src, month_src=month_src, day_src=day_src,
        month_marker=month_marker, tgt_kind=kind, ideal=ideal,
    )


def build_alignments(model: Seq2Seq, dataset: D.DateDataset, indices,
                     max_len: int | None = None, verbose: bool = False) -> list[Alignment]:
    indices = list(indices)
    rows = []
    for n, i in enumerate(indices):
        rows.append(build_alignment(model, dataset, i, max_len))
        if verbose and (n + 1) % 50 == 0:
            print(f"      已处理 {n + 1}/{len(indices)}")
    return rows


# --------------------------------------------------------------------------------------
# 定量分析
# --------------------------------------------------------------------------------------
def alignment_stats(rows: list[Alignment]) -> dict:
    """对齐质量的定量指标。

    · **argmax 命中率**：跳过 '/'，统计 argmax 落在理想集合里的比例
    · **分类型命中率**：日 / 月 / 年 各自命中率
    · **年份位置精确率**：输出年份第 k 位是否正好看源句年份第 k 位（严格 1-1）
    · **单调性**：相邻两个输出步的 argmax 是否单调**递减**
      （源序是「年→月→日」、目标序是「日→月→年」，方向正好相反）
    · **归一化熵**：按输出类型分组，越小说明注意力越集中
    """
    hit_total = hit_ok = 0
    per_kind = {KIND_DAY: [0, 0], KIND_MONTH: [0, 0], KIND_YEAR: [0, 0]}
    month_digits_only = [0, 0]      # 只算「月」的数字、不含定界符 '月'
    month_on_marker = [0, 0]        # argmax 落在 '月' 上的比例
    month_by_value: dict[int, list[int]] = {}
    year_exact = [0, 0]
    day_exact = [0, 0]
    cross_ok = cross_total = 0      # 不同字段之间：源序 年→月→日，目标序 日→月→年，应递减
    within_ok = within_total = 0    # 同一字段内部：照抄顺序一致，应递增
    entropies = {KIND_DAY: [], KIND_MONTH: [], KIND_YEAR: [],
                 KIND_SEP: [], KIND_EOS: []}
    sep_uniform = [0.0, 0]

    for r in rows:
        am = r.argmax_positions()
        ent = r.row_entropy()
        for t, kind in enumerate(r.tgt_kind):
            entropies.setdefault(kind, []).append(float(ent[t]))
            if kind in (KIND_SEP, KIND_EOS):
                # '/' 不需要看源句（理想是均匀散开）；<eos> 是终止决策，两者都不判对错
                if kind == KIND_SEP:
                    sep_uniform[0] += float(r.weights[t].max())
                    sep_uniform[1] += 1
                continue
            hit_total += 1
            ok = am[t] in r.ideal[t]
            hit_ok += int(ok)
            per_kind[kind][0] += int(ok)
            per_kind[kind][1] += 1
            if kind == KIND_MONTH:
                month_digits_only[1] += 1
                month_digits_only[0] += int(am[t] in set(r.month_src))
                month_on_marker[1] += 1
                month_on_marker[0] += int(am[t] == r.month_marker)
                mv = int("".join(r.src_tokens[i] for i in r.month_src))
                slot = month_by_value.setdefault(mv, [0, 0])
                slot[1] += 1
                slot[0] += int(am[t] == r.month_marker)

        # 年份严格 1-1：输出第 k 位 -> 源句年份第 k 位
        for n, t in enumerate(r.positions_of_kind(KIND_YEAR)):
            if n < len(r.year_src):
                year_exact[1] += 1
                year_exact[0] += int(am[t] == r.year_src[n])
        # 日的数字右对齐比较（目标会去掉前导零）
        day_pos = r.positions_of_kind(KIND_DAY)
        for n, t in enumerate(day_pos):
            si = len(r.day_src) - len(day_pos) + n
            if 0 <= si < len(r.day_src):
                day_exact[1] += 1
                day_exact[0] += int(am[t] == r.day_src[si])

        # 单调性：分「字段间」和「字段内」两种，方向正好相反
        #   字段间：目标从「日」走到「年」，源位置应从大到小  -> 递减
        #   字段内：输出「5960」的四位应依次看源句的四位      -> 递增
        seq = [(t, am[t], r.tgt_kind[t]) for t, k in enumerate(r.tgt_kind)
               if k not in (KIND_SEP, KIND_EOS)]
        for (t0, a0, k0), (t1, a1, k1) in zip(seq, seq[1:]):
            if k0 != k1:
                cross_total += 1
                cross_ok += int(a1 <= a0)
            else:
                within_total += 1
                within_ok += int(a1 >= a0)

    def rate(pair):
        return pair[0] / pair[1] if pair[1] else float("nan")

    return {
        "n_samples": len(rows),
        "argmax_hit": rate([hit_ok, hit_total]),
        "hit_day": rate(per_kind[KIND_DAY]),
        "hit_month": rate(per_kind[KIND_MONTH]),
        "hit_month_digits_only": rate(month_digits_only),
        "month_on_marker": rate(month_on_marker),
        "month_marker_rate_by_value": {
            m: (v[0] / v[1] if v[1] else float("nan"))
            for m, v in sorted(month_by_value.items())
        },
        "hit_year": rate(per_kind[KIND_YEAR]),
        "year_position_exact": rate(year_exact),
        "day_position_exact": rate(day_exact),
        "cross_field_decreasing": rate([cross_ok, cross_total]),
        "within_field_increasing": rate([within_ok, within_total]),
        "sep_max_weight_mean": sep_uniform[0] / sep_uniform[1] if sep_uniform[1] else float("nan"),
        "entropy": {k: (float(np.mean(v)) if v else float("nan"))
                    for k, v in entropies.items()},
        "n_hit": hit_total,
    }


def span_matrix(rows: list[Alignment], kind: str,
                span_len: int | None = None, tgt_len: int | None = None) -> np.ndarray:
    """把某一类输出 token 的注意力，**重新归一化到理想源跨度内**再平均。

    得到的是 `(目标位序, 源跨度位序)` 的小矩阵，两位都是右对齐。
    理想情况下这应该是一个清晰的**对角线** —— 那才说明模型学到的是
    「第 k 位照抄第 k 位」，而不是"大范围地看了这一片"。
    """
    if kind not in (KIND_DAY, KIND_MONTH, KIND_YEAR):
        raise ValueError(f"kind 必须是 day/month/year，收到 {kind!r}")

    spans = [r.field_of_kind(kind) for r in rows if r.positions_of_kind(kind)]
    if not spans:
        return np.zeros((1, 1))
    span_len = span_len or max(len(s) for s in spans)
    tgt_len = tgt_len or max(len(r.positions_of_kind(kind)) for r in rows)

    total = np.zeros((tgt_len, span_len))
    count = np.zeros((tgt_len, span_len))
    for r in rows:
        pos = r.positions_of_kind(kind)
        span = r.field_of_kind(kind)
        if not pos or not span:
            continue
        for n, t in enumerate(pos):
            row = tgt_len - len(pos) + n
            w = r.weights[t][span].astype(float)
            s = w.sum()
            if s <= 0:
                continue
            w = w / s                                  # 在跨度内重新归一化
            for m, val in enumerate(w):
                col = span_len - len(span) + m
                total[row, col] += val
                count[row, col] += 1
    return total / np.maximum(count, 1)


def canonical_average_weights(rows: list[Alignment]) -> tuple[np.ndarray, np.ndarray]:
    """把所有样本的注意力映射到**固定格子**上再平均。

    返回 `(mean_weights (9,11), counts (9,11))`。

    目标格子（9 行）：`[日① 日②] / 月份 / [年① 年② 年③ 年④]`（日的数字右对齐）
    源格子（11 列）：`[年①..年④] 年 [月① 月②] 月 [日① 日②] 日`（月和日右对齐）

    这样不同长度的样本才能叠加在同一张热力图上。
    """
    nt, ns = len(CANONICAL_TARGET_LABELS), len(CANONICAL_SOURCE_LABELS)
    total = np.zeros((nt, ns))
    count = np.zeros((nt, ns))

    for r in rows:
        # 目标行映射（右对齐各个段）
        t_map: dict[int, int] = {}
        day_pos = r.positions_of_kind(KIND_DAY)
        for n, t in enumerate(day_pos):
            t_map[t] = 2 - len(day_pos) + n                   # 0,1
        for t, k in enumerate(r.tgt_kind):
            if k == KIND_SEP:
                t_map[t] = 2 if t < r.positions_of_kind(KIND_MONTH)[0] else 4
            elif k == KIND_MONTH:
                t_map[t] = 3
            # KIND_EOS 没有对应的规范行，跳过
        year_pos = r.positions_of_kind(KIND_YEAR)
        for n, t in enumerate(year_pos):
            t_map[t] = 5 + n                                  # 5..8

        # 源列映射（右对齐）
        # 目标列布局：0-3 年数字、4 '年'、5-6 月数字、7 '月'、8-9 日数字、10 '日'
        # 月份数字要落在 5~6（右对齐到 6）：len=2 -> 5,6；len=1 -> 6
        # 日数字要落在 8~9（右对齐到 9）：len=2 -> 8,9；len=1 -> 9
        s_map: dict[int, int] = {}
        for n, si in enumerate(r.year_src):
            s_map[si] = n                                     # 0..3
        s_map[r.src_tokens.index("年")] = 4
        for n, si in enumerate(r.month_src):
            s_map[si] = 7 - len(r.month_src) + n
        s_map[r.src_tokens.index("月")] = 7
        for n, si in enumerate(r.day_src):
            s_map[si] = 10 - len(r.day_src) + n
        s_map[r.src_tokens.index("日")] = 10

        for t, ti in t_map.items():
            row = r.weights[t]
            for si, ci in s_map.items():
                total[ti, ci] += float(row[si])
                count[ti, ci] += 1

    return total / np.maximum(count, 1), count


# --------------------------------------------------------------------------------------
# 终端渲染
# --------------------------------------------------------------------------------------
_SHADES = " .:-=+*#%@"


def _shade(value: float, vmax: float) -> str:
    if vmax <= 0:
        return " "
    k = int(round(value / vmax * (len(_SHADES) - 1)))
    return _SHADES[max(0, min(k, len(_SHADES) - 1))]


def render_ascii(row: Alignment, show_values: bool = False, col_width: int = 5,
                 indent: str = "  ") -> str:
    """把一条样本的注意力矩阵画成终端热力图（CJK 宽度已处理）。

    所有行的前缀都固定为 `pad_display(目标token, 7) + pad_display(类别, 3)` = 10 列，
    这样表头和数据行的格子才会严格对齐（CJK 是双宽字符，不能按字符数算）。
    """
    lines: list[str] = []
    nt, ns = row.weights.shape
    vmax = float(row.weights.max()) if row.weights.size else 0.0
    prefix_w = 10

    lines.append(f"{indent}{row.src_text}   →   {row.tgt_text}")
    tail = " " * 2                                  # 与数据行的「✓」标记字段对齐
    lines.append(indent + " " * prefix_w
                 + "".join(pad_display(str(i), col_width) for i in range(ns)) + tail)
    lines.append(indent + pad_display("源", prefix_w)
                 + "".join(pad_display(t, col_width) for t in row.src_tokens) + tail)
    lines.append(indent + "-" * prefix_w + "-" * (col_width * ns) + "-" * 2)

    am = row.argmax_positions()
    for t in range(nt):
        cells = "".join(pad_display(_shade(row.weights[t, s], vmax), col_width)
                        for s in range(ns))
        kind = row.tgt_kind[t]
        flag = ""
        if kind not in (KIND_SEP, KIND_EOS):
            flag = "✓" if am[t] in row.ideal[t] else "✗"
        extra = ""
        if show_values:
            extra = "  " + " ".join(f"{row.weights[t, s]:.2f}" for s in range(ns))
        lines.append(f"{indent}{pad_display(row.tgt_tokens[t], 7)}"
                     f"{pad_display(KIND_CN[kind], 3)}"
                     f"{cells}{pad_display(flag, 2)}{extra}")
    lines.append(f"{indent}{' ' * prefix_w}"
                 f"（✓ = argmax 落在理想源位置内；止 = <eos> 那一步，不判对错）")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# PNG 渲染（Pillow）
# --------------------------------------------------------------------------------------
def _load_font(size: int):
    from PIL import ImageFont
    for path in CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size), path
            except Exception:  # pragma: no cover
                continue
    return ImageFont.load_default(), None


def _heat_color(v: float) -> tuple[int, int, int]:
    """白 -> 蓝 的连续色带（v 已归一到 [0,1]）。"""
    v = float(min(max(v, 0.0), 1.0))
    # 白 (255,255,255) -> 深蓝 (8,48,107)
    lo = np.array([255, 255, 255], dtype=float)
    hi = np.array([8, 48, 107], dtype=float)
    c = lo + (hi - lo) * (v ** 0.75)
    return tuple(int(round(x)) for x in c)


def save_png(panels, path, title: str = "", cell_w: int = 46, cell_h: int = 34,
             label_w: int = 96, top_h: int = 74, font_size: int = 17,
             panel_gap: int = 34, subtitle_size: int = 15):
    """把若干热力图面板画成一张 PNG。

    参数
    ----
    panels : list[dict]，每个面板含
        {"weights": (T,S), "row_labels": [...], "col_labels": [...],
         "subtitle": str, "annotate": bool}

    返回 (宽, 高, 字体路径)。字体路径为 None 表示没找到支持中文的字体
    （这时中文会显示成方框，请改用终端 ASCII 版）。
    """
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("panels 不能为空")

    font, font_path = _load_font(font_size)
    font_small, _ = _load_font(subtitle_size)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def text_w(s: str, f) -> float:
        if not s:
            return 0.0
        try:
            return float(probe.textlength(s, font=f))
        except Exception:                                  # pragma: no cover
            return len(s) * font_size * 0.62

    head_h, col_h, pad = 24, 40, 20
    # 行标签可能比默认宽度长（比如 "November"、"<eos>"），按实际需要扩宽
    longest_label = max(
        (max(p["row_labels"], key=lambda s: text_w(s, font)) for p in panels
         if p["row_labels"]),
        key=lambda s: text_w(s, font), default="",
    )
    label_w = max(label_w, int(text_w(longest_label, font)) + 10)
    widths = [pad + label_w + cell_w * p["weights"].shape[1] + pad for p in panels]
    heights = [head_h + col_h + cell_h * p["weights"].shape[0] for p in panels]
    if title:
        widths.append(pad + text_w(title, font) + pad)
    for p in panels:
        if p.get("subtitle"):
            widths.append(pad + text_w(p["subtitle"], font_small) + pad)
        for j, c in enumerate(p["col_labels"]):
            widths.append(pad + label_w + cell_w * j + max(text_w(str(c), font_small), 12) + 8)

    canvas_w = int(max(60, max(widths)))
    canvas_h = int(top_h + sum(heights) + panel_gap * (len(panels) - 1) + 24)

    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if title:
        draw.text((pad, 16), title, fill=(20, 20, 20), font=font)

    y = top_h
    for p, panel_h in zip(panels, heights):
        w = np.asarray(p["weights"], dtype=float)
        rows = list(p["row_labels"])
        cols = list(p["col_labels"])
        vmax = float(w.max()) if w.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

        if p.get("subtitle"):
            draw.text((pad, y), p["subtitle"], fill=(70, 70, 70), font=font_small)
        y0 = y + head_h

        # 列标签：索引 + token
        for j, c in enumerate(cols):
            cx = pad + label_w + j * cell_w + cell_w // 2
            draw.text((cx - text_w(str(j), font_small) / 2, y0),
                      str(j), fill=(120, 120, 120), font=font_small)
            draw.text((cx - text_w(str(c), font_small) / 2, y0 + 16),
                      str(c), fill=(40, 40, 40), font=font_small)

        y1 = y0 + col_h
        for i in range(w.shape[0]):
            label = str(rows[i])
            draw.text((pad + label_w - 8 - text_w(label, font),
                       y1 + i * cell_h + cell_h // 2 - font_size // 2),
                      label, fill=(40, 40, 40), font=font)
            for j in range(w.shape[1]):
                x0 = pad + label_w + j * cell_w
                yy0 = y1 + i * cell_h
                draw.rectangle([x0, yy0, x0 + cell_w - 2, yy0 + cell_h - 2],
                               fill=_heat_color(w[i, j] / vmax), outline=(215, 215, 215))
                if p.get("annotate") and cell_w >= 34 and w[i, j] >= 0.005:
                    txt = f"{w[i, j]:.2f}"
                    dark = w[i, j] / vmax > 0.55
                    draw.text((x0 + 4, yy0 + cell_h // 2 - font_size // 2), txt,
                              fill=(255, 255, 255) if dark else (60, 60, 60),
                              font=font_small)
        y += panel_h + panel_gap

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return img.size[0], img.size[1], font_path
