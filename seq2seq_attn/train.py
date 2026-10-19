"""训练循环与评估。

评估分两套指标，这是本步最有价值的设计：

    teacher forcing 评估：第 t 步喂**真实的** y_{t-1}
                          快，但数字系统性偏高（推理时拿不到真实上文）
    自回归贪心评估      ：第 t 步喂**模型自己生成的** ŷ_{t-1}
                          这才是真实使用场景，也是第 6 步的主题

两者的差距就是 **exposure bias** 的量化体现。
"""

from __future__ import annotations

import dataclasses
import random
import time

import numpy as np

from . import data as D
from .config import CONFIG, Config
from .model import Seq2Seq, greedy_decode
from .optim import Adam, clip_grad_global_norm, collect_params, warmup_cosine_lr

__all__ = [
    "build_model",
    "sanity_check_overfit",
    "evaluate_teacher_forcing",
    "evaluate_greedy",
    "decode_sentences",
    "train",
]


# --------------------------------------------------------------------------------------
# 建模
# --------------------------------------------------------------------------------------
def build_model(train_set: D.DateDataset, cfg_model) -> Seq2Seq:
    """按训练集词表造模型（词表只来自训练集，保持评估诚实）。"""
    return Seq2Seq.build(
        cfg_model,
        src_vocab_size=len(train_set.src_vocab),
        tgt_vocab_size=len(train_set.tgt_vocab),
        src_pad_id=train_set.src_vocab.pad_id,
        tgt_pad_id=train_set.tgt_vocab.pad_id,
        seed=0,
    )


def _make_optimizer(model: Seq2Seq, cfg_train) -> tuple[dict, dict, Adam]:
    params, grads = collect_params(model)
    opt = Adam(params, grads, lr=cfg_train.lr, beta1=cfg_train.beta1,
               beta2=cfg_train.beta2, eps=cfg_train.eps)
    return params, grads, opt


# --------------------------------------------------------------------------------------
# 批次流
# --------------------------------------------------------------------------------------
def _batch_stream(dataset: D.DateDataset, batch_size: int, seed: int, drop_last: bool = True):
    """无限批次生成器：每个 epoch 重新打乱一次。

    为什么要 drop_last：最后一个不完整批次样本少，算出来的平均梯度噪声更大；
    固定大小能让 loss 曲线更平滑、更容易看出趋势。
    """
    rng = random.Random(seed)
    while True:
        batches = D.make_batches(dataset, batch_size, shuffle=True, rng=rng, drop_last=drop_last)
        for batch in batches:
            yield batch


# --------------------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------------------
def evaluate_teacher_forcing(model: Seq2Seq, dataset: D.DateDataset,
                             batch_size: int = 64, max_batches: int | None = None) -> dict:
    """教师强制评估：token 准确率 + **整句完全正确率**。

    两个都要看：token 准确率 99% 而整句正确率 0% 是完全可能的
    （每句只错一个字符，整句就算错）。
    """
    batches = D.make_batches(dataset, batch_size, shuffle=False)
    if max_batches is not None:
        batches = batches[:max_batches]

    n_tok = n_tok_correct = 0
    n_seq = n_seq_correct = 0
    total_loss = 0.0

    for batch in batches:
        h_seq, h_final = model.enc.forward(batch.src, batch.src_mask)
        logits, _ = model.dec.forward(batch.tgt_in, h_seq, batch.src_mask, h_final)
        loss = model.loss.forward(logits, batch.tgt_out, batch.tgt_mask)
        total_loss += loss * batch.batch_size

        pred = logits.argmax(axis=-1)                       # (B,T)
        mask = batch.tgt_mask > 0.5
        n_tok += int(mask.sum())
        n_tok_correct += int(((pred == batch.tgt_out) & mask).sum())

        row_ok = np.all(np.where(mask, pred == batch.tgt_out, True), axis=1)
        n_seq += batch.batch_size
        n_seq_correct += int(row_ok.sum())

    return {
        "loss": total_loss / max(n_seq, 1),
        "token_acc": n_tok_correct / max(n_tok, 1),
        "seq_acc": n_seq_correct / max(n_seq, 1),
        "n_seq": n_seq,
    }


def evaluate_greedy(model: Seq2Seq, dataset: D.DateDataset, max_len: int,
                    batch_size: int = 64, max_batches: int | None = None) -> dict:
    """自回归贪心评估：模型只能看它自己生成的上文。"""
    batches = D.make_batches(dataset, batch_size, shuffle=False)
    if max_batches is not None:
        batches = batches[:max_batches]

    pad_id = dataset.tgt_vocab.pad_id
    eos_id = dataset.tgt_vocab.eos_id
    bos_id = dataset.tgt_vocab.bos_id

    n_tok = n_tok_correct = 0
    n_seq = n_seq_correct = 0

    for batch in batches:
        seqs, _, _ = greedy_decode(model, batch.src, batch.src_mask, max_len,
                                   bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                                   collect_weights=False)
        for i in range(batch.batch_size):
            ref = [int(v) for v in batch.tgt_out[i] if int(v) != pad_id]
            hyp = [int(v) for v in seqs[i] if int(v) != pad_id]
            m = min(len(ref), len(hyp))
            n_tok += len(ref)
            n_tok_correct += sum(1 for j in range(m) if ref[j] == hyp[j])
            n_seq += 1
            if ref == hyp:
                n_seq_correct += 1

    return {
        "token_acc": n_tok_correct / max(n_tok, 1),
        "seq_acc": n_seq_correct / max(n_seq, 1),
        "n_seq": n_seq,
    }


def decode_sentences(model: Seq2Seq, dataset: D.DateDataset, indices, max_len: int) -> list[dict]:
    """把指定样本的「参考译文 / teacher forcing 预测 / 自回归贪心预测」都解出来。"""
    sv, tv = dataset.src_vocab, dataset.tgt_vocab
    indices = list(indices)
    batch = D.collate(dataset, indices)

    h_seq, h_final = model.enc.forward(batch.src, batch.src_mask)
    logits, weights = model.dec.forward(batch.tgt_in, h_seq, batch.src_mask, h_final)
    tf_pred = logits.argmax(axis=-1)

    seqs, _, _ = greedy_decode(model, batch.src, batch.src_mask, max_len,
                               bos_id=tv.bos_id, eos_id=tv.eos_id, pad_id=tv.pad_id)

    out = []
    for i, idx in enumerate(indices):
        ex = dataset.examples[idx]
        tf_ids = [int(v) for v in tf_pred[i] if int(v) != tv.pad_id]
        hyp_ids = [int(v) for v in seqs[i] if int(v) != tv.pad_id]
        out.append({
            "src": ex.src_text,
            "ref": ex.tgt_text,
            "tf": _render(tv, tf_ids),
            "greedy": _render(tv, hyp_ids),
            "weight": weights[i].copy(),
            "src_tokens": list(ex.src_tokens),
            "greedy_ids": hyp_ids,
        })
    return out


def _render(vocab: D.Vocab, ids) -> str:
    """把 id 序列渲染成可读的英文日期（去掉 <eos> 之后的内容）。"""
    tokens = []
    for i in ids:
        token = vocab.itos[int(i)]
        if token == "<eos>":
            break
        if token in D.SPECIAL_TOKENS:
            continue
        tokens.append(token)
    return "".join(tokens)


# --------------------------------------------------------------------------------------
# 健全性检查
# --------------------------------------------------------------------------------------
def sanity_check_overfit(cfg: Config, n_examples: int = 10, steps: int = 400,
                         lr: float | None = None, seed: int = 0,
                         verbose: bool = True) -> dict:
    """**先过拟合一小撮样本**：这是最省时间的排错手段。

    如果模型连 10 条样本都记不住（loss 降不到接近 0），那一定是代码有 bug，
    而不是"模型太小""数据太难"—— 这种情况下调超参数纯粹是浪费时间。

    它同时验证三件独立的事：
        1. 前向能产生有区分度的输出
        2. 反向能把梯度正确传回**每一个**参数
        3. 优化器真的在更新参数
    """
    train, _, _ = D.build_datasets(cfg.data)
    small = train.subset(n_examples)
    model = build_model(train, cfg.model)
    params, grads, opt = _make_optimizer(model, cfg.train)
    if lr is not None:
        opt.set_lr(lr)

    batch = D.collate(small, list(range(len(small))))
    history: list[float] = []
    acc_history: list[float] = []

    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        loss, _, _ = model.forward(batch)
        model.backward()
        clip_grad_global_norm(params, grads, cfg.train.grad_clip)
        opt.step()
        history.append(loss)
        acc_history.append(model.last_accuracy)
        if verbose and (step % max(1, steps // 8) == 0 or step == steps - 1):
            print(f"      step {step:>4}   loss {loss:.6f}   token_acc {model.last_accuracy:.4f}")
    elapsed = time.perf_counter() - t0

    final = evaluate_teacher_forcing(model, small, batch_size=len(small))
    greedy = evaluate_greedy(model, small, max_len=small.max_tgt_len + 2, batch_size=len(small))

    return {
        "model": model,
        "dataset": small,
        "batch": batch,
        "loss_history": history,
        "acc_history": acc_history,
        "first_loss": history[0],
        "final_loss": history[-1],
        "final": final,
        "greedy": greedy,
        "elapsed": elapsed,
        "steps": steps,
    }


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------
def train(cfg: Config = CONFIG, *, max_steps: int | None = None, run_name: str = "base",
          seed: int | None = None, verbose: bool = True,
          eval_greedy: bool = True, log_every: int | None = None,
          eval_every: int | None = None, snapshot_at=None) -> dict:
    """完整训练流程。

    每个 step 的顺序（顺序很重要）：
        1. opt.zero_grad()           —— 清零，否则梯度会跨 step 累积
        2. loss = model.forward(b)   —— 前向
        3. model.backward()          —— 反向（只累加）
        4. 记录梯度范数              —— **裁剪之前**记录，否则看不到爆炸
        5. clip_grad_global_norm()   —— 裁剪
        6. opt.step()                —— 更新参数

    snapshot_at: 在这些 step 处保存一份参数副本（第 6 步要用不同训练程度的
                 checkpoint 来对比解码策略，一次训练就能拿到多个快照）。
    """
    seed = cfg.train.seed if seed is None else seed
    max_steps = max_steps or cfg.train.max_steps
    log_every = log_every or cfg.train.log_every
    eval_every = eval_every or cfg.train.eval_every
    snapshot_at = set(snapshot_at or ())
    snapshots: dict[int, dict] = {}

    train_set, dev_set, test_set = D.build_datasets(cfg.data)
    model = build_model(train_set, cfg.model)
    params, grads, opt = _make_optimizer(model, cfg.train)

    stream = _batch_stream(train_set, cfg.data.batch_size, seed=seed)
    max_len = max(train_set.max_tgt_len, dev_set.max_tgt_len, test_set.max_tgt_len) + 2

    history = {
        "step": [], "loss": [], "lr": [], "grad_norm": [], "train_acc": [],
        "eval_step": [], "dev_loss": [], "dev_tf_token": [], "dev_tf_seq": [],
        "dev_ar_token": [], "dev_ar_seq": [],
    }

    if verbose:
        print(f"  模型：{model}")
        print(f"  参数总量 {model.num_params():,}"
              f"（编码器 {model.enc.num_params():,} + 解码器 {model.dec.num_params():,}）")
        print(f"  训练集 {len(train_set)} / 开发集 {len(dev_set)} / 测试集 {len(test_set)}")
        print(f"  最大解码步数 max_len = {max_len}")
        print(f"  计划 {max_steps} 步，batch_size={cfg.data.batch_size}")

    t0 = time.perf_counter()
    for step in range(max_steps):
        lr = warmup_cosine_lr(step, max_steps, cfg.train.lr, warmup_frac=0.1, min_lr_ratio=0.1)
        opt.set_lr(lr)

        batch = next(stream)
        opt.zero_grad()
        loss, _, _ = model.forward(batch)
        model.backward()
        grad_norm = clip_grad_global_norm(params, grads, cfg.train.grad_clip)
        opt.step()

        history["step"].append(step)
        history["loss"].append(loss)
        history["lr"].append(lr)
        history["grad_norm"].append(grad_norm)
        history["train_acc"].append(model.last_accuracy)

        if (step + 1) in snapshot_at:
            snapshots[step + 1] = model.state_dict()

        if verbose and (step % log_every == 0 or step == max_steps - 1):
            print(f"    step {step:>5}/{max_steps}   loss {loss:.4f}   "
                  f"acc {model.last_accuracy:.4f}   lr {lr:.2e}   "
                  f"|g| {grad_norm:7.3f}   {time.perf_counter() - t0:6.1f}s")

        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            dev = evaluate_teacher_forcing(model, dev_set)
            history["eval_step"].append(step)
            history["dev_loss"].append(dev["loss"])
            history["dev_tf_token"].append(dev["token_acc"])
            history["dev_tf_seq"].append(dev["seq_acc"])
            if eval_greedy:
                ar = evaluate_greedy(model, dev_set, max_len)
                history["dev_ar_token"].append(ar["token_acc"])
                history["dev_ar_seq"].append(ar["seq_acc"])
            else:
                history["dev_ar_token"].append(float("nan"))
                history["dev_ar_seq"].append(float("nan"))
            if verbose:
                extra = ""
                if eval_greedy:
                    extra = (f"  |  自回归 token {ar['token_acc']:.4f}"
                             f"  seq {ar['seq_acc']:.4f}")
                print(f"      [dev] loss {dev['loss']:.4f}  "
                      f"TF token {dev['token_acc']:.4f}  seq {dev['seq_acc']:.4f}{extra}")

    elapsed = time.perf_counter() - t0

    result = {
        "model": model,
        "optimizer": opt,
        "history": history,
        "train_set": train_set,
        "dev_set": dev_set,
        "test_set": test_set,
        "max_len": max_len,
        "elapsed": elapsed,
        "run_name": run_name,
        "steps": max_steps,
        "snapshots": snapshots,
    }

    if verbose:
        print(f"\n  训练用时 {elapsed:.1f}s（{elapsed / max_steps * 1000:.1f} ms/step）")
        print("\n  最终评估：")
        print(f"    {'数据集':<10}{'TF loss':>10}{'TF token':>11}{'TF seq':>10}"
              f"{'AR token':>11}{'AR seq':>10}")
        print("    " + "-" * 62)
        for name, ds in (("train", train_set), ("dev", dev_set), ("test", test_set)):
            tf = evaluate_teacher_forcing(model, ds)
            ar = evaluate_greedy(model, ds, max_len)
            result[f"{name}_tf"] = tf
            result[f"{name}_ar"] = ar
            print(f"    {name:<10}{tf['loss']:>10.4f}{tf['token_acc']:>11.4f}"
                  f"{tf['seq_acc']:>10.4f}{ar['token_acc']:>11.4f}{ar['seq_acc']:>10.4f}")
        gap = result["test_tf"]["seq_acc"] - result["test_ar"]["seq_acc"]
        print(f"\n    exposure bias（test 上 TF seq − AR seq）= {gap:+.4f}")
        if gap > 0.02:
            print("    → TF 明显高于自回归：模型依赖了推理时拿不到的真实上文。")
        print("\n  样例（测试集里的未见过年份）：")
        for row in decode_sentences(model, test_set, range(8), max_len):
            mark = "✓" if row["ref"] == row["greedy"] else "✗"
            print(f"    {row['src']:<18} 参考 {row['ref']:<18} "
                  f"TF {row['tf']:<18} 贪心 {row['greedy']:<18} {mark}")

    return result
