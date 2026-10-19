"""从零手写（纯 NumPy）的带注意力机制序列翻译模型。

子模块：
    config     超参数
    data       语料生成 / 词表 / padding / batch                 （第 1 步 ✅）
    nn         参数容器 / softmax / log_softmax / Linear / Embedding （第 2、3 步 ✅）
    attention  加性注意力 + 缩放点积注意力（前向 + 反向）         （第 2 步 ✅）
    encoder    GRUCell + GRUEncoder（含 BPTT 反向）               （第 3 步 ✅）
    decoder    AttentionDecoder（Bahdanau 式，前向 + 反向）       （第 4 步 ✅）
    losses     CrossEntropyLoss（log_softmax + padding mask）    （第 5 步 ✅）
    optim      Adam / 全局范数裁剪 / warmup+余弦退火             （第 5 步 ✅）
    model      Seq2Seq 组装 + 贪心解码                            （第 5 步 ✅）
    train      训练循环 / 评估 / 健全性检查                       （第 5 步 ✅）
    infer      Beam Search + 暴力枚举参照                         （第 6 步 ✅）
    viz        注意力对齐可视化 + 理想对齐定量分析               （第 7 步 ✅）

消融结论（第 8 步，训练集 500 / 4000 条，test AR 整句准确率）：
    baseline（加性注意力）        0.9950 / 1.0000
    去掉注意力（固定 context）     0.1425 / 0.5975   <- 注意力是必需的
    缩放点积（零参数）            1.0000 / 1.0000   <- 机制比算法细节重要
    输出层只吃 s_t                0.5850 / 0.9325   <- context 直接进输出层很重要
    初始状态恒为零                1.0000 / 1.0000   <- 桥接投影冗余

训练结果（3000 步、206 秒、257,435 参数、纯 NumPy）：
    train / dev / test 的整句准确率都是 100%，其中 test 用的是
    训练时从未出现过的年份（5900~5999）。背诵检测显示模型生成的年份
    0% 落在训练年份集合里 —— 它学到的是「照抄数字」这条规则。

对齐质量（400 条测试样本、2676 个对齐点）：
    argmax 命中率 100%、年份严格位置精确率 100%、
    字段间递减 100%、字段内递增 100%。
    年份跨度矩阵对角线均值 0.9822 —— 学到的是「第 k 位照抄第 k 位」。

⚠ 手写循环网络踩过的坑（都已修好并有回归测试）：
   1. `Module._cache` 只有一个槽位。循环里被调用多次的模块
      （解码器里有 attn / cell / out 三个）必须逐步 `save_cache()` /
      `load_cache()`，否则梯度完全错误 —— 但形状全对、不报错、不出 nan。
   2. `Embedding.backward` 必须用 `np.add.at`：numpy 花式索引是拷贝语义，
      重复 token（日期里的 '1' 出现 4 次）会丢梯度。
   3. 反向传播里的汇合点必须相加：`context` 用在两处、`s_t` 用在两处、
      `dkeys + dvalues` 来自同一份 `H_seq`。
   4. 梯度检验的单一阈值有适用边界。`attn.W_q` 的梯度比 loss 小 8 个数量级，
      逐元素相对误差必然很大 —— 判据改用自校准形式（见 nn.check_gradient）。
   5. 数据划分太窄会让模型靠背诵过关。最初 train_years=(2000,2029)
      每年前后出现 133 次，模型背下 30 个年份，测试集整句准确率 0%。
      现在由 tests/test_data.py 断言保证「每个数位的数字都覆盖」
      和「样本数/不同年份数 接近 1」。
   6. Beam Search 不能因为「已完成假设攒够 beam_size 条」就提前停止。
      `done` 里够数不代表存活 beam 里没有更好的候选 —— 实测会让
      100% 的模型掉到 60%。正确做法：跑到 max_len，
      最后从「已完成 ∪ 仍存活」里一起挑。
   7. 评价指标的定义错误看起来和模型错误一模一样。第 7 步的理想对齐
      漏写了 '月' 定界符，把「只在 10 月关注定界符」这个完全正确的
      行为误判成 10% 失败率。先看失败样本有什么共同点。
   8. 梯度检验要用**真实的训练损失**。第 8 步用未 mask 的人造损失
      `(logits*G).sum()` 时，`Embedding(padding_idx=0)` 会把 <pad> 行
      的梯度强制置零（真实 CE 用 tgt_mask 已经置零了，所以无害），
      两者不一致 -> 误报失败。另外判据的容差必须带噪声地板的**绝对项**，
      否则 spread=0 且梯度≈0 的元素会把容差压到 1e-300 而假阳性。
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["config", "data", "nn", "attention", "encoder", "decoder",
           "losses", "optim", "model", "train", "infer", "viz", "__version__"]
