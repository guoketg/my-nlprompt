# NLPrompt Contest — 流水线命令与各阶段结果（PIPELINE）

> 返回 [主控 HANDOVER](../HANDOVER.md)

## 5. V2 训练/推理命令（当前主线）

### 5.1 训练（LoRA 微调）
```bash
cd /root/code/NLPrompt
setsid nohup ./.venv/bin/python -u train_clip_lora.py \
    --data-root /root/datasets/contest \
    --json all_class_predictions.json \
    --proto-dir output/contest_prototype \
    --output-dir output/contest_ft_lora \
    --epochs 40 --warmup-epochs 5 --update-interval 5 \
    --batch-size 64 --resolution 224 \
    --lora-lr 1e-4 --head-lr 1e-3 \
    --retention-ratio 0.8 --proto-keep-ratio 0.5 \
    --keep-best-val \
    > /tmp/lora_train.log 2>&1 < /dev/null &
```
- 关键超参：LoRA rank=8, alpha=16, dropout=0.05，目标层 `out_proj,c_fc,c_proj`；分类头 `Linear(512,500)` + CE；动态筛选 = 类内 loss 最小 top-0.8 ∩ 原型相似度 top-0.5，迭代重训。
- **提交模型选择**：用 `--keep-best-val`（保存 max-val_acc epoch，即 warmup 期，线上 55% 来源）；**不要用 `--keep-final`**（保存末轮，阶段 A 实测线上仅 34%）。
- 日志：`tail -f /tmp/lora_train.log`

### 5.2 推理（出提交包）
```bash
cd /root/code/NLPrompt
setsid nohup ./.venv/bin/python -u test_clip_lora.py \
    --ckpt output/contest_ft_lora/best.pt \
    --output-dir output/contest_ft_lora \
    --data-root /root/datasets/contest \
    > /tmp/lora_test.log 2>&1 < /dev/null &
```
- 输出 `output/contest_ft_lora/pred_results.{csv,zip}`（5-view TTA：中心 + 角落 + 翻转，soft-vote）。
- 增强 TTA 用法：`test_clip_lora.py --tta`（11-view）/ `--tta2`（14-view）。
- 校验格式见 [03-pitfalls §4.1](03-pitfalls.md#41-提交格式无表头--4-位零填充)。

### 5.3 自训练（self_train_v2.py）
```bash
# c2 激进自训练（已出 67.86%）
./.venv/bin/python -u self_train_v2.py \
  --seed-ckpt output/contest_ft_lora_c/best.pt \
  --output-dir output/contest_ft_lora_c2 \
  --rounds 5 --consistent-conf-threshold 0.7 --mismatch-conf-threshold 0.8 \
  --drop-mismatch --use-uncertain

# 软标签自训练（方向 B，进行中，见 07-next §12）
./.venv/bin/python -u self_train_v2.py \
  --seed-ckpt output/contest_ft_lora_c2/best.pt \
  --output-dir output/contest_ft_lora_c2_soft \
  --rounds 5 --consistent-conf-threshold 0.75 --mismatch-conf-threshold 0.9 \
  --use-uncertain --drop-mismatch --soft-labels --soft-temp 1.0
```

### 5.4 概率导出与融合（export_c2_probs_swa.py）
```bash
# 导出 6 检查点 11-view TTA probs
./.venv/bin/python -u export_c2_probs_swa.py --export --data-root /root/datasets/contest
# 等权融合 -> output/fuse_swa_all6/
./.venv/bin/python -u export_c2_probs_swa.py --fuse all
# 14-view 版（tta2）
./.venv/bin/python -u export_c2_probs_swa.py --export --tta2 --data-root /root/datasets/contest
./.venv/bin/python -u export_c2_probs_swa.py --fuse all --tta2
```

## 6. 各阶段结果归档

### 6.1 阶段 B（✅ 完成 2026-08-10）
- 提交品：`output/contest_ft_lora_b/pred_results.zip`，**线上 63.66%**（+8.64pp vs 旧基线 55.02%）。
- 训练配置：全量 103,218 候选、40 epoch、warmup 5、best.pt=epoch5（val_acc 0.772）。
- 训练耗时：约 5.6 小时。关键：已超 PGDF 同骨干 59%。

### 6.2 阶段 C 自训练（✅ 完成）
- 种子：`output/contest_ft_lora_b/best.pt`（63.66%）。
- 核心：predict(M) → consistency_clean → 净化集重训 → 迭代 3 round。
- 耗时：约 7.4 小时。提交品：`output/contest_ft_lora_c/pred_results.zip`，**线上 64.66%**（+1.0pp vs 阶段 B）。

### 6.3 阶段 C2 激进自训练（✅ 完成 2026-08-11）
- 种子：`output/contest_ft_lora_c/best.pt`（64.66%）。
- 核心：predict(M) → consistency_clean(mismatch 直接剔除) → 净化集重训 → 迭代 **5 轮**；conf 阈值 0.7/0.8。
- 命令：`self_train_v2.py --seed-ckpt output/contest_ft_lora_c/best.pt --output-dir output/contest_ft_lora_c2 --rounds 5 --consistent-conf-threshold 0.7 --mismatch-conf-threshold 0.8 --drop-mismatch --use-uncertain`
- 出包：增强 TTA（11-view）。提交品：`output/contest_ft_lora_c2/pred_results.zip`，**线上 67.86%**（+3.20pp vs 阶段 C）。
- 关键数据：Round1 mismatch=5157 / uncertain=25269 → Round2 mismatch=3025（自训练正循环持续生效）；Round5 val_acc=0.9335。

### 6.4 方向 4：三重信号串联自训练（❌ 已证伪 2026-08-12）
- 动机：C2 一致性信号已饱和，需独立信号（loss+proto+一致性）打破 confirmation bias。
- 实测：2 轮 × 12 epoch，线上 **64.65%**，比基线 68.37% 低 3.72pp。
- 证伪原因：过度去噪，砍掉困难真样本。详见 [06-learnings §11.6](06-learnings.md#116-方向-4-证伪自训练去噪存在最优阈值2026-08-12)。**本任务不再试信号串联**。
