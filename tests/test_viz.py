"""注意力对齐解析、定量分析与渲染的断言测试。

运行：
    python tests/test_viz.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

import numpy as np

from seq2seq_attn import data as D
from seq2seq_attn import train as T
from seq2seq_attn import viz as V
from seq2seq_attn.config import CONFIG

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


# --------------------------------------------------------------------------------------
def test_display_width() -> None:
    print("\n== CJK 显示宽度 ==")
    check("ASCII 宽度 = 字符数", V.display_width("abc") == 3)
    check("汉字算两列", V.display_width("年月日") == 6)
    check("混排", V.display_width("2024年") == 6, str(V.display_width("2024年")))
    check("pad_display 左对齐", V.display_width(V.pad_display("年", 6)) == 6)
    check("pad_display 右对齐", V.pad_display("年", 6, "right").endswith("年"))
    check("pad_display 不截断超长文本", V.pad_display("abcdefgh", 3) == "abcdefgh")


def test_parse_source_and_target() -> None:
    print("\n== 源句 / 目标句解析 ==")
    check("无补零源句",
          V._parse_source(["5", "9", "6", "0", "年", "6", "月", "2", "2", "日"])
          == ([0, 1, 2, 3], [5], [7, 8]))
    check("补零源句",
          V._parse_source(["5", "9", "6", "0", "年", "0", "6", "月", "0", "5", "日"])
          == ([0, 1, 2, 3], [5, 6], [8, 9]))
    check("两位月份",
          V._parse_source(["5", "9", "6", "0", "年", "1", "0", "月", "2", "2", "日"])
          == ([0, 1, 2, 3], [5, 6], [8, 9]))

    check("目标切分", V._split_target(["2", "2", "/", "June", "/", "5", "9", "6", "0"])
          == (2, 1, 4))
    check("单位数日", V._split_target(["5", "/", "May", "/", "5", "9", "6", "0"]) == (1, 1, 4))

    for label, fn, arg in [
        ("源句缺 '日'", V._parse_source, ["5", "年", "6", "月"]),
        ("源句顺序错", V._parse_source, ["日", "年", "月"]),
        ("源句月段为空", V._parse_source, ["5", "年", "月", "2", "日"]),
        ("目标只有一个 '/'", V._split_target, ["2", "/", "June"]),
        ("目标月份不是 1 个 token", V._split_target, ["2", "/", "J", "u", "n", "e", "/", "5"]),
    ]:
        try:
            fn(arg)
            check(f"拒绝：{label}", False)
        except ValueError:
            check(f"拒绝：{label}", True)


def test_alignment_shape_validation() -> None:
    print("\n== Alignment 形状校验 ==")
    ok = V.Alignment(index=0, src_text="", tgt_text="",
                     src_tokens=["a", "b"], tgt_tokens=["x"],
                     weights=np.ones((1, 2)))
    check("形状匹配时正常构造", ok.src_len == 2 and ok.weights.shape == (1, 2))
    for label, w in [("列数多了", np.ones((1, 3))), ("行数少了", np.ones((0, 2)))]:
        try:
            V.Alignment(index=0, src_text="", tgt_text="", src_tokens=["a", "b"],
                        tgt_tokens=["x"], weights=w)
            check(f"拒绝：{label}", False)
        except ValueError:
            check(f"拒绝：{label}", True)
    try:
        V.Alignment(index=0, src_text="", tgt_text="", src_tokens=["a"],
                    tgt_tokens=["x"], weights=np.ones(2))
        check("拒绝：权重不是二维", False)
    except ValueError:
        check("拒绝：权重不是二维", True)


# --------------------------------------------------------------------------------------
def _model_and_sets():
    train_set, dev_set, test_set = D.build_datasets(CONFIG.data)
    model = T.build_model(train_set, CONFIG.model)
    return model, train_set, dev_set, test_set


def _perfect_weights(row: V.Alignment) -> np.ndarray:
    """按理想对齐造一个「完美」的注意力矩阵（每行 one-hot）。"""
    w = np.zeros_like(row.weights)
    n_sep = n_eos = 0
    for t, kind in enumerate(row.tgt_kind):
        if kind == V.KIND_SEP:
            w[t] = 1.0 / w.shape[1]                     # 均匀散开
            n_sep += 1
        elif kind == V.KIND_EOS:
            w[t] = 1.0 / w.shape[1]
            n_eos += 1
        elif kind == V.KIND_YEAR:
            n = row.tgt_kind[:t].count(V.KIND_YEAR)
            w[t, row.year_src[n]] = 1.0
        elif kind == V.KIND_DAY:
            pos = row.positions_of_kind(V.KIND_DAY)
            n = pos.index(t)
            w[t, row.day_src[len(row.day_src) - len(pos) + n]] = 1.0
        else:                                            # month
            w[t, row.month_marker] = 1.0
    return w


def test_build_alignment() -> None:
    print("\n== build_alignment ==")
    model, train_set, _, test_set = _model_and_sets()
    row = V.build_alignment(model, test_set, 0)

    check("权重形状 = (解码步数, 真实源长)",
          row.weights.shape == (len(row.tgt_tokens), len(row.src_tokens)),
          str(row.weights.shape))
    check("目标 token 末尾补了 <eos>", row.tgt_tokens[-1] == "<eos>")
    check("类别数 = 解码步数", len(row.tgt_kind) == row.weights.shape[0])
    check("理想集合数 = 解码步数", len(row.ideal) == row.weights.shape[0])
    check("'/' 的理想集合为空",
          all(not row.ideal[t] for t, k in enumerate(row.tgt_kind) if k == V.KIND_SEP))
    check("<eos> 的理想集合为空", not row.ideal[-1])
    check("月份的理想集合含数字和 '月'",
          row.ideal[row.positions_of_kind(V.KIND_MONTH)[0]]
          == set(row.month_src) | {row.month_marker})
    check("年份的理想集合 = 源句年份四位",
          row.ideal[row.positions_of_kind(V.KIND_YEAR)[0]] == set(row.year_src))
    check("每行权重和为 1", close(row.weights.sum(axis=1), 1.0, tol=1e-12))
    check("行熵在 [0,1]", bool(np.all((row.row_entropy() >= -1e-12) &
                                      (row.row_entropy() <= 1 + 1e-12))))

    # 补零写法的源句也要解析正确
    padded = [i for i in range(len(test_set))
              if len(test_set.examples[i].src_tokens) == 11
              and test_set.examples[i].src_tokens[5] == "0"]
    if padded:
        r2 = V.build_alignment(model, test_set, padded[0])
        check("补零样本的月段含前导零", len(r2.month_src) == 2, str(r2.month_src))
        check("补零样本权重形状正确",
              r2.weights.shape == (len(r2.tgt_tokens), 11), str(r2.weights.shape))
    else:
        check("（测试集中没有补零样本，跳过）", True)

    # 批量接口
    rows = V.build_alignments(model, test_set, range(5))
    check("build_alignments 返回 5 条", len(rows) == 5)
    check("build_alignments 接受生成器", len(V.build_alignments(model, test_set, iter(range(3)))) == 3)


def test_alignment_stats_on_perfect_and_broken() -> None:
    print("\n== 对齐统计：完美 / 打乱 / 错位 ==")
    model, _, _, test_set = _model_and_sets()
    rows = V.build_alignments(model, test_set, range(12))

    perfect = []
    for r in rows:
        r2 = V.Alignment(**{**r.__dict__, "weights": _perfect_weights(r)})
        perfect.append(r2)
    st = V.alignment_stats(perfect)
    check("完美对齐：argmax 命中率 = 1", abs(st["argmax_hit"] - 1.0) < 1e-12, f"{st['argmax_hit']}")
    check("完美对齐：日/月/年各自命中率 = 1",
          abs(st["hit_day"] - 1) < 1e-12 and abs(st["hit_month"] - 1) < 1e-12
          and abs(st["hit_year"] - 1) < 1e-12)
    check("完美对齐：年份位置精确率 = 1", abs(st["year_position_exact"] - 1) < 1e-12)
    check("完美对齐：日的右对齐精确率 = 1", abs(st["day_position_exact"] - 1) < 1e-12)
    check("完美对齐：字段间递减 = 1", abs(st["cross_field_decreasing"] - 1) < 1e-12)
    check("完美对齐：字段内递增 = 1", abs(st["within_field_increasing"] - 1) < 1e-12)
    check("完美对齐：查源句的三类熵 ≈ 0",
          st["entropy"]["day"] < 1e-9 and st["entropy"]["year"] < 1e-9
          and st["entropy"]["month"] < 1e-9)
    check("完美对齐：'/' 熵 ≈ 1（均匀）",
          abs(st["entropy"]["sep"] - 1.0) < 1e-9, f"{st['entropy']['sep']}")
    check("完美对齐：'/' 最大权重 ≈ 1/源长",
          abs(st["sep_max_weight_mean"] - 1.0 / 11) < 0.05,
          f"{st['sep_max_weight_mean']:.4f}")

    # 随机权重 —— 命中率应该接近「逐个理想集合大小 / 源长」的期望
    rng = np.random.default_rng(0)
    random_rows = [V.Alignment(**{**r.__dict__, "weights": rng.dirichlet(
        np.ones(r.weights.shape[1]), size=r.weights.shape[0])}) for r in rows]
    st_r = V.alignment_stats(random_rows)
    check("随机权重：命中率远低于 1", st_r["argmax_hit"] < 0.6, f"{st_r['argmax_hit']:.4f}")
    check("随机权重：命中率高于 0", st_r["argmax_hit"] > 0.0)
    check("随机权重：熵明显高于完美对齐",
          st_r["entropy"]["year"] > st["entropy"]["year"] + 0.3)

    # 整体平移一位 —— 集合口径只掉一半（字段内平移仍落在同一个理想集合里），
    # 但**严格位置精确率**会崩掉。这正是两个指标都要看的原因。
    shifted = []
    for r in rows:
        w = np.zeros_like(r.weights)
        src = _perfect_weights(r)
        w[:, 1:] = src[:, :-1]
        w[:, 0] = src[:, -1]
        shifted.append(V.Alignment(**{**r.__dict__, "weights": w}))
    st_s = V.alignment_stats(shifted)
    check("整体错位后「集合命中率」只部分下降（对字段内错位不敏感）",
          0.3 < st_s["argmax_hit"] < 0.8, f"{st_s['argmax_hit']:.4f}")
    check("整体错位后「严格位置精确率」崩掉",
          st_s["year_position_exact"] < 0.2, f"{st_s['year_position_exact']:.4f}")
    check("统一平移**不**破坏单调性（相对顺序没变）",
          abs(st_s["within_field_increasing"] - 1.0) < 1e-12,
          f"{st_s['within_field_increasing']:.4f}")

    # 字段内**逆序** —— 仍落在理想集合里（集合命中率不掉），
    # 但破坏了「第 k 位照抄第 k 位」，严格位置和单调性都该崩。
    reversed_rows = []
    for r in rows:
        w = np.zeros_like(r.weights)
        for kind in (V.KIND_YEAR, V.KIND_DAY):
            pos = r.positions_of_kind(kind)
            span = r.span_of_kind(kind)
            for n, t in enumerate(pos):
                m = len(pos) - 1 - n
                if m < len(span):
                    w[t, span[m]] = 1.0
        for t, kind in enumerate(r.tgt_kind):
            if kind in (V.KIND_SEP, V.KIND_EOS):
                w[t] = 1.0 / w.shape[1]
            elif kind == V.KIND_MONTH:
                w[t, r.month_marker] = 1.0
        reversed_rows.append(V.Alignment(**{**r.__dict__, "weights": w}))
    st_rev = V.alignment_stats(reversed_rows)
    check("字段内逆序后集合命中率仍为 1（都还在理想集合内）",
          abs(st_rev["argmax_hit"] - 1.0) < 1e-12, f"{st_rev['argmax_hit']:.4f}")
    check("字段内逆序后「字段内递增」崩掉",
          st_rev["within_field_increasing"] < 0.2, f"{st_rev['within_field_increasing']:.4f}")
    check("字段内逆序后「严格位置精确率」也崩掉",
          st_rev["year_position_exact"] < 0.2, f"{st_rev['year_position_exact']:.4f}")

    # 空输入
    st_e = V.alignment_stats([])
    check("空输入不崩且返回 nan", st_e["n_samples"] == 0 and st_e["argmax_hit"] != st_e["argmax_hit"])


def test_span_matrix() -> None:
    print("\n== 跨度矩阵 ==")
    model, _, _, test_set = _model_and_sets()
    rows = V.build_alignments(model, test_set, range(20))

    perfect = [V.Alignment(**{**r.__dict__, "weights": _perfect_weights(r)}) for r in rows]

    M = V.span_matrix(perfect, "year", span_len=4, tgt_len=4)
    check("完美对齐的年份跨度矩阵 = 单位阵", close(M, np.eye(4), tol=1e-12), str(M.round(3)))

    Md = V.span_matrix(perfect, "day", span_len=2, tgt_len=2)
    check("完美对齐的日跨度矩阵对角占优",
          float(np.mean(np.diag(Md))) > 0.99 and float(Md[0, 1] + Md[1, 0]) < 0.02,
          str(Md.round(3)))

    Mm = V.span_matrix(perfect, "month", span_len=3, tgt_len=1)
    check("完美对齐的月份峰值在 '月'（第 3 列）", int(np.argmax(Mm[0])) == 2, str(Mm.round(3)))

    # 真实（未训练）模型：注意力接近均匀，跨度矩阵也应该是均匀的
    Mr = V.span_matrix(rows, "year", span_len=4, tgt_len=4)
    check("未训练模型的跨度矩阵值域合理",
          bool(np.all((Mr >= -1e-12) & (Mr <= 1 + 1e-12))), str(Mr.round(3)))
    check("未训练模型的年份跨度矩阵接近均匀（≈1/4）",
          float(np.std(Mr)) < 0.02 and abs(float(Mr.mean()) - 0.25) < 0.02,
          f"mean={Mr.mean():.4f} std={Mr.std():.4f}")
    check("未训练模型的行和为 1",
          close(Mr.sum(axis=1), 1.0, tol=1e-9), str(Mr.sum(axis=1).round(6)))

    check("span_matrix 支持空输入", V.span_matrix([], "year").shape == (1, 1))
    try:
        V.span_matrix(rows, "nope")
        check("拒绝未知 kind", False)
    except ValueError:
        check("拒绝未知 kind", True)


def test_canonical_average() -> None:
    print("\n== 规范化平均矩阵 ==")
    model, _, _, test_set = _model_and_sets()
    rows = V.build_alignments(model, test_set, range(30))

    # ---- 结构性质用「合成完美对齐」来测（未训练模型的注意力接近均匀，argmax 是任意的）
    perfect = [V.Alignment(**{**r.__dict__, "weights": _perfect_weights(r)}) for r in rows]
    M, C = V.canonical_average_weights(perfect)

    check("形状固定为 (9, 11)", M.shape == (9, 11), str(M.shape))
    check("标签数与之匹配",
          len(V.CANONICAL_TARGET_LABELS) == 9 and len(V.CANONICAL_SOURCE_LABELS) == 11)
    check("有数据的格子覆盖大部分", int((C > 0).sum()) >= 60, str(int((C > 0).sum())))
    check("权重值域合理", bool(np.all((M >= -1e-12) & (M <= 1 + 1e-12))))
    check("日的两行峰值在右侧（第 8、9 列）",
          int(np.argmax(M[0])) == 8 and int(np.argmax(M[1])) == 9,
          f"行0={int(np.argmax(M[0]))} 行1={int(np.argmax(M[1]))}")
    check("月份行峰值在第 7 列（'月'）", int(np.argmax(M[3])) == 7,
          f"{int(np.argmax(M[3]))}")
    check("年份四行峰值依次在 0、1、2、3 列",
          [int(np.argmax(M[i])) for i in range(5, 9)] == [0, 1, 2, 3],
          str([int(np.argmax(M[i])) for i in range(5, 9)]))
    # '/' 的理想权重是"均匀散开"。注意不同样本源长是 9/10/11，
    # 均匀值分别是 1/9、1/10、1/11，平均后落在 0.091~0.111 之间。
    for t, label in ((2, "第一个 '/'"), (4, "第二个 '/'")):
        check(f"{label} 行所有列都有权重（散开）",
              bool(np.all(M[t] > 0.08)), f"min={M[t].min():.4f}")
        check(f"{label} 行没有聚焦（最大值 < 0.12）",
              float(M[t].max()) < 0.12, f"max={M[t].max():.4f}")
        check(f"{label} 行近似均匀（极差 < 0.03）",
              float(M[t].max() - M[t].min()) < 0.03,
              f"极差={M[t].max() - M[t].min():.4f}")

    # ---- 未训练模型：只测接口与覆盖，不测峰值位置
    Mr, Cr = V.canonical_average_weights(rows)
    check("未训练模型也能映射到同样形状", Mr.shape == (9, 11), str(Mr.shape))
    check("未训练模型的覆盖格数一致", int((Cr > 0).sum()) == int((C > 0).sum()))
    check("未训练模型的值接近均匀",
          float(np.std(Mr)) < 0.05, f"std={np.std(Mr):.4f}")

    check("空输入返回零矩阵", V.canonical_average_weights([])[0].shape == (9, 11))


def test_render_ascii() -> None:
    print("\n== 终端渲染 ==")
    model, _, _, test_set = _model_and_sets()
    row = V.build_alignment(model, test_set, 0)
    text = V.render_ascii(row)
    lines = text.split("\n")

    check("行数 = 头部 4 行 + 解码步数 + 尾注 1 行",
          len(lines) == 4 + len(row.tgt_tokens) + 1, str(len(lines)))
    check("包含源句文本", row.src_text in text)
    check("包含目标句文本", row.tgt_text in text)
    check("包含每个源 token",
          all(t in text for t in row.src_tokens))
    check("包含 <eos> 那一行", "<eos>" in text)
    # 表头（索引/源 token/分隔线）与数据行的格子必须从同一列开始
    body = lines[1 : 4 + len(row.tgt_tokens)]
    check("表头与数据行的显示宽度完全一致（CJK 对齐）",
          len({V.display_width(l) for l in body}) == 1,
          str(sorted({V.display_width(l) for l in body})))
    check("分隔线的长度与数据行一致",
          V.display_width(lines[3]) == V.display_width(lines[4]))

    text_v = V.render_ascii(row, show_values=True)
    w = row.weights
    check("show_values 时每行都带上权重数值",
          all(f"{w[0, j]:.2f}" in text_v for j in range(w.shape[1])),
          f"首行前 3 个值 {[f'{w[0, j]:.2f}' for j in range(3)]}")
    check("show_values 比默认更长", V.display_width(text_v.split("\n")[4]) >
          V.display_width(lines[4]))


def test_save_png() -> None:
    print("\n== PNG 渲染 ==")
    model, _, _, test_set = _model_and_sets()
    row = V.build_alignment(model, test_set, 0)
    panel = {"weights": row.weights, "row_labels": list(row.tgt_tokens),
             "col_labels": list(row.src_tokens), "subtitle": "测试", "annotate": True}

    # 写到工作区内的目录（不用 tempfile：沙箱下清理临时目录会被拒绝访问）
    out = pathlib.Path(__file__).resolve().parents[1] / "viz" / "_test"
    out.mkdir(parents=True, exist_ok=True)
    try:
        p = out / "a.png"
        w, h, font = V.save_png([panel], p, title="标题")
        check("文件已生成", p.exists() and p.stat().st_size > 0, f"{p.stat().st_size} B")
        check("尺寸合理", w > 100 and h > 100, f"{w}×{h}")
        check("返回字体路径（找不到时应为 None）", font is None or pathlib.Path(font).exists(),
              str(font))
        from PIL import Image
        with Image.open(p) as im:
            check("PNG 可被 Pillow 打开且尺寸一致", im.size == (w, h), str(im.size))

        p2 = out / "nested" / "b.png"
        V.save_png([panel, panel], p2, title="两个面板")
        check("支持多层目录自动创建", p2.exists() and p2.stat().st_size > 0)

        try:
            V.save_png([], out / "empty.png")
            check("拒绝空 panels", False)
        except ValueError:
            check("拒绝空 panels", True)

        # 行列标签比默认宽度长时不能溢出
        long_panel = {"weights": np.ones((2, 3)) * 0.3,
                      "row_labels": ["November", "<eos>"],
                      "col_labels": ["10", "月", "日"],
                      "subtitle": "一个相当长的副标题用来测试画布宽度是否够" * 2,
                      "annotate": True}
        p3 = out / "long.png"
        w3, h3, _ = V.save_png([long_panel], p3, title="长标题" * 8)
        check("长标签 / 长标题时画布够宽", w3 > 400, f"{w3}×{h3}")
        check("长标签面板的宽度不是由格子数决定的",
              w3 > 20 + 96 + 3 * 46 + 20, f"{w3}")
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    check("测试目录已清理", not out.exists())


def test_real_model_alignment_quality() -> None:
    print("\n== 真实模型（未训练）也能跑通全流程 ==")
    model, train_set, _, test_set = _model_and_sets()
    rows = V.build_alignments(model, test_set, range(16))
    st = V.alignment_stats(rows)
    check("统计字段齐全",
          {"n_samples", "argmax_hit", "hit_day", "hit_month", "hit_year",
           "year_position_exact", "day_position_exact", "cross_field_decreasing",
           "within_field_increasing", "sep_max_weight_mean", "entropy",
           "month_marker_rate_by_value"} <= set(st), str(sorted(st)))
    check("n_samples 正确", st["n_samples"] == 16)
    check("各命中率都在 [0,1]",
          all(0.0 <= st[k] <= 1.0 for k in
              ("argmax_hit", "hit_day", "hit_month", "hit_year",
               "year_position_exact", "day_position_exact",
               "cross_field_decreasing", "within_field_increasing")))
    check("未训练模型的命中率不会高得离谱", st["argmax_hit"] < 0.9, f"{st['argmax_hit']:.4f}")
    # 16 条随机样本只会覆盖到一部分月份，这里只检查表结构合理
    check("月份关注定界符的比例表非空且键在 1~12",
          len(st["month_marker_rate_by_value"]) >= 5
          and all(1 <= m <= 12 for m in st["month_marker_rate_by_value"]),
          str(sorted(st["month_marker_rate_by_value"])))
    check("比例值都在 [0,1]",
          all(0.0 <= v <= 1.0 for v in st["month_marker_rate_by_value"].values()))
    check("日的/年的严格位置精确率在 [0,1]",
          0.0 <= st["day_position_exact"] <= 1.0 and 0.0 <= st["year_position_exact"] <= 1.0)


def main() -> None:
    print("=" * 70)
    print(" 注意力可视化与分析 测试")
    print("=" * 70)
    test_display_width()
    test_parse_source_and_target()
    test_alignment_shape_validation()
    test_build_alignment()
    test_alignment_stats_on_perfect_and_broken()
    test_span_matrix()
    test_canonical_average()
    test_render_ascii()
    test_save_png()
    test_real_model_alignment_quality()
    print("\n" + "=" * 70)
    print(f" 全部通过：{PASSED} 项断言 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
