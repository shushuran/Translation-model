"""集中管理全部超参数。

把超参数和代码分开，后面做消融实验（比如换注意力类型、改数据划分）时
只需要 `dataclasses.replace(CONFIG.model, attn_type="dot")`，不用改动任何模型代码。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "Config",
    "CONFIG",
]


@dataclass(frozen=True)
class DataConfig:
    """数据管道相关配置。"""

    # 三个划分的样本数
    n_train: int = 4000
    n_dev: int = 400
    n_test: int = 400
    batch_size: int = 32

    # 中文里「3月」写成「03月」的概率。给数据加一点真实的书写噪声，
    # 逼着模型去判断「哪一个数字才是要输出的那一位」，而不是死记位置。
    pad_variant_prob: float = 0.5

    # 是否在日期后附加星期（如「2024年3月5日星期二」→「5/March/2024/Tuesday」）。
    # 打开后任务更难、注意力对齐更丰富，可作为后续挑战。
    include_weekday: bool = False

    # 按年份划分数据集：测试年份在训练时完全没见过。
    # 因为数字是逐位切分的，这些年份的数字位都在词表里，
    # 所以这考查的是「组合泛化」而不是「认不认识的词」。
    #
    # ⚠ 划分区间不能太窄 —— 这是第 5 步训练时用实测数据纠正过来的：
    #   最初用 train=(2000,2029)（30 个年份）+ 4000 条样本，
    #   等于每个年份平均出现 133 次，模型直接把这 30 个年份背下来了
    #   （训练集整句准确率 97.8%，测试集 0%，而且生成的年份**从不超出**
    #     训练年份集合）。同时 2000~2029 的年份第三位只能是 0/1/2，
    #   测试年份的第三位是 3 —— 这个「位置上的数字」训练时根本没见过。
    #
    #   现在的划分满足两条硬性要求（由 tests/test_data.py 断言保证）：
    #     1. 每个数位上出现过的数字，训练集都要覆盖到
    #        （测试年份 59xx 的千位 5、百位 9、十位/个位 0~9 全都在训练集里出现）
    #     2. 「样本数 / 不同年份数」接近 1，让背诵不可能奏效
    train_years: tuple[int, int] = (1000, 5799)
    dev_years: tuple[int, int] = (5800, 5899)
    test_years: tuple[int, int] = (5900, 5999)

    seed: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """模型结构配置（第 3、4 步会用到）。"""

    src_emb_dim: int = 64
    tgt_emb_dim: int = 64
    hidden_dim: int = 128          # GRU 隐状态维度
    attn_dim: int = 128            # 加性注意力中间层维度

    # "bahdanau" = 加性注意力（Bahdanau et al., 2015）
    # "dot"      = 缩放点积注意力（Vaswani et al., 2017）
    # "mean"     = 无注意力基线：context 恒为编码器输出的掩码平均
    #              （即经典 Seq2Seq 的"固定 context 向量"方案，
    #                第 8 步用它量化注意力的贡献）
    # 三种都手写，方便做对比。
    attn_type: str = "bahdanau"

    # 编码器是否双向。默认关闭：注意力本身已经能「看到全序列」，
    # 双向作为第 8 步的扩展项，避免一开始就把反向传播写复杂。
    bidirectional: bool = False

    # ---- 解码器（第 4 步）----
    # 输出层（投影到词表）吃什么：
    #   "s_c"      -> [s_t ; c_t]                （推荐，最简洁）
    #   "s_c_emb"  -> [s_t ; c_t ; emb(y_{t-1})] （Bahdanau 原论文的输入）
    #   "s"        -> [s_t]                      （消融基线：context 只能经由隐状态影响输出）
    decoder_output: str = "s_c"

    # 解码器初始隐状态 s_{-1} 从哪来：
    #   "bridge" -> s_0 = tanh(W·h_final + b)   （可学投影，推荐）
    #   "direct" -> s_0 = h_final               （零参数，要求两边隐维度相同）
    #   "zero"   -> s_0 = 0                     （消融基线：全靠注意力提供源句信息）
    decoder_init: str = "bridge"

    init_scale: float = 0.08       # 参数初始化 uniform(-scale, scale)


@dataclass(frozen=True)
class TrainConfig:
    """训练配置（第 5 步会用到）。"""

    max_steps: int = 3000
    lr: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 5.0
    teacher_forcing: float = 1.0   # 1.0 = 一直用真实上一个词
    log_every: int = 200
    eval_every: int = 200
    seed: int = 0


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# 全局默认配置。想改实验条件时用 replace() 派生新配置，不要原地修改。
CONFIG = Config()
