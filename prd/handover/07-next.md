# NLPrompt Contest — 后续方向（NEXT）

> 返回 [主控 HANDOVER](../HANDOVER.md)

## 12. 方向 B：软标签自训练（已验收证伪，2026-08-14）

### 结论
软标签自训练出包 **67.08% / 66.49%**（11-view TTA），低于 c2 单模 67.86% / 融合 68.95%，**方向不成立**。完整根因与对比见 [06-learnings §11.10](06-learnings.md#1110-方向-b软标签证伪2026-08-14)。

### 已实现代码（self_train_v2.py，保留备用）
- `predict_all` 额外返回全量 softmax 分布 `probs_all` (n_total × 500) 作为软标签目标。
- `--soft-labels` / `--soft-temp`：开启后用 `probs_all` 作 soft target（可温度平滑），训练 loss 从 CE 切换为 KL 蒸馏。
- 实验配置：`--seed-ckpt output/contest_ft_lora_c2/best.pt --output-dir output/contest_ft_lora_c2_soft --rounds 5 --consistent-conf-threshold 0.75 --mismatch-conf-threshold 0.9 --use-uncertain --drop-mismatch --soft-labels --soft-temp 1.0`。

### 教训（泛化）
在"类内噪声严重 + 无类别名 + 细粒度"结构下，**既不可硬删困难样本（§11.6），也不可软标困难样本（§11.10）**；confirmation bias 不会被平滑目标消除。剩余训练侧杠杆只剩方向 C（难样本加权不删）。

## 13. 任务收尾与后续（长尾 + 噪声）

### 13.1 方向 C：难样本加权不删（待启动）

**动机（直击 §11.6 + §11.10 共同根因）**：
- §11.6（v3 串联去噪）证伪 = 硬删困难样本砍掉最有信息量监督；§11.10（软标签）证伪 = 软标困难样本稀释最有信息量监督。
- **方向 C 的核心差异**：保留全部样本、不对困难样本做"删除"或"软化"，而是按**模型一致性/置信度给样本级权重**——困难样本（低置信 / folder≠pred）权重降低但**仍参与训练**，易样本正常权重。这样既不注入错误硬标签、也不丢信息，只是让模型"少听"噪声、"多听"干净信号。这是 §11.6/§11.10 之后的最后一个"不砍困难样本监督"的杠杆。

**与现有代码的关系（最小改动方案）**：
- `self_train_v2.py` 已有 `sample_weights` 机制：当前 c2 模式 `sw[mismatch]=0.1, sw[uncertain]=0.05`，但配合 `--drop-mismatch` 时 mismatch 直接不进训练集。
- 方向 C = **去掉 `--drop-mismatch`，且把 mismatch/uncertain 权重从 0.1/0.05 抬到更接近 1.0**（例如 `--mismatch-weight 0.6 --uncertain-weight 0.4`），即"全量保留 + 温和降权"。这是与 c2（激进剔除）最干净的对照变量。

**计划新增参数（self_train_v2.py）**：
- `--mismatch-weight`（默认 0.1，方向 C 设为 0.6~0.9）
- `--uncertain-weight`（默认 0.05，方向 C 设为 0.3~0.5）
- 语义：当 `--drop-mismatch` 未设时，`sw[mismatch]=args.mismatch_weight, sw[uncertain]=args.uncertain_weight`，所有样本进 `train_mask`。

**实验配置（待启动，公平对比 c2）**：
```bash
./.venv/bin/python -u self_train_v2.py \
  --seed-ckpt output/contest_ft_lora_c2/best.pt \
  --output-dir output/contest_ft_lora_c3_hardweight \
  --rounds 5 --consistent-conf-threshold 0.75 --mismatch-conf-threshold 0.9 \
  --use-uncertain --mismatch-weight 0.7 --uncertain-weight 0.4
# 注意：不加 --drop-mismatch，全量保留
```
- 种子同 c2（`contest_ft_lora_c2/best.pt`），其余与 c2 同构，**唯一变量 = 去噪激进度（删 vs 加权）**。
- 验收同 c2：11-view TTA 公平对比 67.86% / 68.95%。
  - 若 > c2 → 确认"难样本加权不删"是有效新杠杆，可叠加 tta2 + 加轮次深入，并刷新最优记录 70.04%。
  - 若 ≤ c2 → 训练侧在噪声结构下已穷尽，定格 70.04% 等下月「长尾 + 噪声」数据集。

**决策状态**：用户 2026-08-14 确认执行方向 C（[HANDOVER §0](../HANDOVER.md)）。代码改动与启动命令待 AI 执行。


**本任务定位（2026-08-12 明确）**：NLPrompt 比赛是**噪声标签鲁棒方法的探索沙盒**，目的是为后续「长尾分布 + 标签噪声」真实场景积累方法，而非刷分。当前最优 **70.04%**（`fuse_swa_all6_tta2`）作为方法记录。

**噪声标签方法沉淀（可迁移到长尾+噪声）**：
1. **全量候选池去噪**（B 阶段）：+8.64pp —— 对长尾「尾部类样本少易误删」尤其关键，需改「按类配额保留」。
2. **伪标签一致性自训练**（C/C2）：+1.0pp；注意 C2 的"单信号甜点"教训（[06-learnings §11.6](06-learnings.md#116-方向-4-证伪自训练去噪存在最优阈值2026-08-12)）：去噪阈值过狠会砍困难样本，长尾下尾部困难样本占比更高，阈值须更保守。
3. **同线多快照 SWA 融合**（[06-learnings §11.8](06-learnings.md#118-同线-6-快照-swa-融合6895--6946+051pp已验证-2026-08-12)）：零成本 +0.51pp，可直接迁移。
4. **11-view/14-view TTA**：零成本 +0.3~0.6pp，可直接迁移。
5. **已证伪**：三重信号串联去噪（[06-learnings §11.6](06-learnings.md#116-方向-4-证伪自训练去噪存在最优阈值2026-08-12)）、多模型集成、提分辨率、zero-shot 清洗（无类别名）。

**长尾 + 噪声 待探究方向（新任务起点，另起仓库）**：类平衡采样 / 类级损失加权 / 尾部类原型增强；去噪与长尾的耦合（尾部类噪声更难识别）；混合训练时注入长尾先验。
