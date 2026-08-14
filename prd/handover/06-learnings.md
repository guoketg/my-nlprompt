# NLPrompt Contest — 经验总结（LEARNINGS）

> 返回 [主控 HANDOVER](../HANDOVER.md)

## 11.1 阶段 B：全量候选池 → 55.02% → 63.66%（+8.64pp，已验证）

**做了什么**：`train_clip_lora.py` 新增 `--candidate full`，候选池从 stage-1 clean_mask 的 54.8%（56,565 张）扩到全量 103,218 张。动态筛选（小损失 + 原型相似度）每 5 个 epoch 从全量重新挑出最干净样本。

**为什么生效**：
1. **stage-1 K-means 去噪是信息瓶颈**：基于冻结 CLIP 特征做静态 K-means 去噪，保留 54.8%。但这 54.8% 里仍混噪声，同时被砍掉的 45.2% 里大量好样本被误杀——因为冻结特征与 LoRA 微调后的分布有偏移。
2. **动态筛选比静态筛选更适配微调分布**：全量候选让动态筛选每 5 个 epoch 重新评估全量样本的 loss+proto，随着模型变好，每次都能从全量挑出当前最干净的样本。"在线去噪" vs "离线去噪"的本质优势。
3. **双信号联合（loss+proto）比单信号稳健**：PGDF 的纯 loss 或纯 proto 更容易被噪声误导，而 loss+proto 交集让两个信号互相验证。
4. **warmup 提交是必要条件**：epoch5 best.pt（val_acc 0.772）出 63.66%，若用末轮会因过拟合筛选子集而暴跌（阶段 A 仅 34%）。`--keep-best-val` 默认化是关键保险。
5. **不要迷信本地 val_acc**：最高 0.772 与线上 63.66% 无直接映射，唯一可信 = 官网提交分。

**教训**：静态 K-means 去噪对微调是信息瓶颈；动态筛选必须搭配全量候选池。

## 11.2 后续方向判断（63.66% 基线）
- 阶段 C（伪标签一致性净化迭代自训练）：预期 **63.66%→65–67%**。这是约束内最后一个还能大幅提分的杠杆。
- TTA 强化 / 融合兜底：边际收益，阶段 C 后考虑。

## 11.3 阶段 C：伪标签一致性自训练 → 63.66% → 64.66%（+1.0pp，已验证）

**做了什么**：`self_train_v2.py` 用阶段 B 的 63.66% 模型作种子，对全量 103k 训练图预测 → 伪标签一致性净化（folder==pred 且 conf≥0.8 = 干净；folder≠pred 且 conf≥0.9 = 错图/错标签，降权 0.1）→ 净化集重训 20 epoch → 迭代 3 轮。

**实测关键数据**（来自 `output/contest_ft_lora_c/self_train_log.json`）：
- Round 1：clean=80,517 / mismatch=11,031 / uncertain=11,670，best_val=0.9622
- Round 2：clean=86,740 / mismatch=6,470 / uncertain=9,669，best_val=0.9783
- Round 3：clean=89,581 / mismatch=4,058 / uncertain=9,579，best_val=**0.9886**
- 趋势：随迭代，clean↑ mismatch↓ → 净化越来越干净，自训练正循环成立。

**为什么有效但增益小于阶段 B（+1.0pp vs +8.64pp）**：
1. 阶段 C 是"锦上添花"：阶段 B 已拿到最大收益。
2. 伪标签一致性是"保守去噪"：大量低置信样本被暂存未用；错图只降权不剔除。
3. 本地 val_acc 0.9886 与线上 64.66% 严重不匹配：再次印证本地指标虚假。

**教训**：自训练正循环已验证成立（clean↑ mismatch↓），可继续加轮次（5 轮）或放松 conf 阈值（0.8→0.7）。不要被本地 val_acc 误导。

## 11.4 阶段 C2：激进自训练 → 64.66% → 67.86%（+3.20pp，已验证 2026-08-11）

**做了什么**：在阶段 C 基础上，把"保守降权"改为"激进剔除"——`--drop-mismatch`（高置信 mismatch 直接删除）+ `--use-uncertain`（低置信样本 0.05 权重参与）+ 轮次 3→5 + conf 阈值 0.8/0.9→0.7/0.8。出包用增强 TTA（11-view）。

**实测关键数据**（来自 `output/contest_ft_lora_c2/self_train_log.json`）：
- Round 1：clean=72,792 / mismatch=5,157 / uncertain=25,269
- Round 2：clean=78,678 / mismatch=3,025 / uncertain=21,515（正循环成立）
- Round 5：val_acc=0.9335
- 训练集规模 ~98k–100k

**为什么增益（+3.20pp）远超预期（+0.5~1pp）**：
1. 剔除 mismatch 比降权更彻底：c 阶段 mismatch 仅降权 0.1 仍在训练，模型被迫拟合错标签样本；c2 直接删除，避免错误信息注入。
2. 放松阈值让更多样本进入训练：uncertain ~25k 参与。
3. 增强 TTA 独立贡献：11-view 比 5-view 更鲁棒。
4. 5 轮迭代让正循环更充分。

**教训**：自训练"激进度"是关键杠杆；本地 val_acc 仍不映射线上；增强 TTA 零训练成本、必带。

## 11.5 检查点融合：67.86% → 68.37%（+0.51pp，已验证 2026-08-11）

**做了什么**：用 `fuse_predictions.py` 把 c2（67.86%）与 c（64.66%）两个检查点的 TTA 概率做 per-class 置信度加权融合，输出 `output/fuse_c2_c/pred_results.zip`。

**结果**：线上 **68.37%**，比更强的单模 c2 再 +0.51pp。

**为什么有效**：c 与 c2 虽同骨干同流程，但经过不同轮次的自训练净化，**错误模式不完全重叠**——c2 在激进剔除中误杀的类，c 反而保留了监督信号。概率平均把互补性兑现为收益。

**合规性**：同一 CLIP ViT-B/32 骨干、同一训练流程的不同检查点做概率平均，等价于"权重滑动平均/快照集成"，属单模型范畴，不违规。

**教训**：融合是低成本兜底，必做；融合收益随两个检查点分差扩大而衰减。

## 11.6 方向 4 证伪：自训练去噪存在最优阈值（2026-08-12）

**结论**：三重信号（loss+proto+一致性）串联自训练 → 单模 **64.65%**，比 C2 单模(67.86%) 低 3.21pp、比融合基线(68.37%) 低 3.72pp。方向证伪。

**根因**：细粒度分类的训练信号高度依赖**困难样本**。C2 用"高置信预测错"单信号去噪，恰好卡在最优阈值——只删明确错图。v3 引入 loss/proto 信号后，额外删掉的样本里混着大量"模型暂时预测错但确为真样本"的困难图，等于**砍掉最有信息量的监督**，模型过拟合到易样本，泛化下降。

**普适教训**：
- 去噪不是越狠越好：C2 的 dropped≈2837 是甜点，v3 的 dropped≈8483 已过线。
- "独立信号"未必打破 confirmation bias：实测中 loss/proto 对困难样本的判定与一致性高度一致，反而放大去噪偏差。
- 不要再试信号串联。剩余杠杆只剩多检查点融合与 TTA 微调。

## 11.7 扩展融合兜底：68.37% → 68.95%（+0.58pp，已验证 2026-08-12）

**做了什么**：用一次性等权平均脚本融合三个检查点的 TTA 概率 —— `c2/best.pt`（67.86%）、`c2/round4.pt`、`c/best.pt`（64.66%），输出 `output/fuse_c2_r4_c/pred_results.zip`。

**结果**：线上 **68.95%**，比原融合基线 68.37% +0.58pp、比 d×c2×c(68.43%) +0.52pp，为当时最优提交。

**为什么比 d 版更强**：d 单模仅 64.65%（证伪模型），拉进三方融合是"弱模型拖累 + 微弱互补"的净 +0.06pp；而 round4 是 c2 训练线里 67.86% 演化路径上的独立检查点（与 best 互补但不弱），替换 d 后改了 8.70% 预测，互补性显著更强且无拖累。

**复现命令**（三方等权平均）：
```python
pa=np.load('output/contest_ft_lora_c2/pred_probs.npy')
pb=np.load('output/contest_ft_lora_c2_round4/pred_probs.npy')
pc=np.load('output/contest_ft_lora_c/pred_probs.npy')
fused=(pa+pb+pc)/3.0
# argmax -> 写 pred_results.csv/zip（names 取自任一 pred_results.csv）
```

**已产出的融合包对比**：
| 包 | 组合 | 线上 |
|----|------|------|
| `fuse_swa_all6_tta2` | c2 线 6 快照等权 (round1~5+best) · 14-view tta2 | **70.04%**（当前最优）|
| `fuse_swa_best40_tta2` | c2 线 best 加权 0.4 · 14-view tta2 | 69.41% |
| `fuse_swa_all6` | c2 线 6 快照等权 · 11-view | 69.46% |
| `fuse_c2_r4_c` | c2 × round4 × c | 68.95% |
| `fuse_swa_best40` | c2 线 best 加权 0.4 | 68.45% |
| `fuse_d_c2_c` | d × c2 × c | 68.43% |
| `fuse_c2_c` | c2 × c | 68.37% |

## 11.8 同线 6 快照 SWA 融合：68.95% → 69.46%（+0.51pp，已验证 2026-08-12）

**做了什么**：导出 c2 训练线全部 6 个检查点（round1.pt~round5.pt + best.pt）的 11-view TTA 概率（各 24967×500），做**等权算术平均**（SWA），输出 `output/fuse_swa_all6/pred_results.zip`。脚本 `export_c2_probs_swa.py`（零训练成本）。

**结果**：线上 **69.46%**，比三线融合 68.95% 再 +0.51pp，刷新当时最优。变体 `fuse_swa_best40`（best 权重 0.4）仅 68.45%——说明**等权反而优于强检查点加权**，early round 的弱检查点并未拖累、反而提供多样互补。

**修正了 §11.7 的假设**：同线早期强检查点可纳入 SWA；跨线的弱/证伪模型（d 64.65%）仍会拖累，勿混入。

**复现命令**：
```bash
./.venv/bin/python -u export_c2_probs_swa.py --export --data-root /root/datasets/contest
./.venv/bin/python -u export_c2_probs_swa.py --fuse all
./.venv/bin/python -u export_c2_probs_swa.py --fuse best5 --alpha 0.4
```

## 11.9 tta2（14-view）增强 TTA + 同线 6 快照 SWA：69.46% → 70.04%（+0.58pp，已验证 2026-08-12）

**做了什么**：把 §11.8 的 TTA 从 11-view (`tta_enhanced`) 升级到 14-view (`tta_enhanced2`，在 11-view 基础上加 `._corner_crops(224)` 四角裁剪共 3 个 transform，合计 14 个）。重新导出 6 检查点 probs（`probs_round1~5_tta2` + `probs_best_tta2`），等权融合 → `output/fuse_swa_all6_tta2/`；best 加权 0.4 → `output/fuse_swa_best40_tta2/`。

**结果**：
- `fuse_swa_all6_tta2` 线上 **70.04%**（新任务最优，比 11-view 同融合 +0.58pp）
- `fuse_swa_best40_tta2` 线上 **69.41%**（仍低于等权 all6）

**结论**：
1. tta2 的 14-view（四角裁剪增强）在**零重训成本**下稳定提 +0.58pp，是本轮最后一项推理侧增益，已并入最优记录 70.04%。
2. 等权 all6 在 11-view 与 14-view 下均优于 best 加权，结论稳健。
3. **推理侧杠杆已用尽**（分辨率/末轮/zero-shot/增强 view 都已试过，边际递减明显）。后续提分必须靠训练侧新方法（见 [07-next §12](07-next.md#12-方向-b软标签自训练进行中)）。

**复现命令**：
```bash
./.venv/bin/python -u export_c2_probs_swa.py --export --tta2 --data-root /root/datasets/contest
./.venv/bin/python -u export_c2_probs_swa.py --fuse all --tta2
./.venv/bin/python -u export_c2_probs_swa.py --fuse best5 --alpha 0.4 --tta2
```

## 11.10 方向 B 软标签证伪（2026-08-14）

**结论**：软标签自训练（KL 蒸馏，soft target = 模型全量 softmax 分布）出包 **67.08% / 66.49%**（11-view TTA 公平对比 c2 单模 67.86% / 融合 68.95%），**低于 c2 线，方向不成立**。

**实验配置**（见 [07-next §12](07-next.md#12-方向-b软标签已验收证伪)）：种子 `contest_ft_lora_c2/best.pt`，5 轮，`--consistent-conf-threshold 0.75 --mismatch-conf-threshold 0.9 --use-uncertain --drop-mismatch --soft-labels --soft-temp 1.0`，`output/contest_ft_lora_c2_soft/`。

**结果对比**：
| 包 | 配置 | 线上 |
|----|------|------|
| `fuse_swa_all6_contest_ft_lora_c2_soft` | 软标签 6 快照等权 · 11-view | 67.08% |
| `fuse_swa_best40_contest_ft_lora_c2_soft` | 软标签 best 加权 0.4 · 11-view | 66.49% |
| `fuse_swa_all6`（c2 基线，对照）| 硬标签 6 快照等权 · 11-view | 69.46% |
| `contest_ft_lora_c2`（c2 单模，对照）| 硬标签 | 67.86% |

**根因分析**：
1. **软标签未能"软化困难样本"反而稀释监督**：软 target 来自自训练模型自身的预测分布，confirmation bias 仍在——错标样本的 softmax 仍集中在错误类。KL 蒸馏让模型去拟合这个带偏分布，等于把噪声"平滑地"灌进训练，没有 hard 丢弃去噪的直接收益，也没有 c2 硬删错图那样干净的净信号。
2. **本数据集噪声结构 ≠ 标准蒸馏友好场景**：c2 已证伪"信号串联去噪"（[§11.6](116-方向-4-证伪自训练去噪存在最优阈值2026-08-12)）证明困难样本=最有信息量监督。软标签对"困难样本"给的是模糊目标，相当于**用低信息量目标替换高信息量目标**，与 §11.6 教训同源——去噪/软化都踩了"砍掉困难样本监督"的同一个坑。
3. **与 §11.6 共同构成普适结论**：在"类内噪声严重 + 无类别名 + 细粒度"结构下，**既不可硬删困难样本（§11.6），也不可软标困难样本（本节）**；唯一验证有效的去噪是 c2 的"高置信预测错单信号硬删"（甜点 dropped≈2837）。

**教训（泛化）**：
- 软标签/蒸馏在非 IID 噪声 + 无 GT 自训练下，confirmation bias 不会被平滑目标消除，只是变隐蔽。
- 不要再试"软化监督目标"类方案（含 label smoothing 蒸馏、EMA 教师软标）。剩余训练侧杠杆只剩**方向 C 难样本加权不删**（保留困难样本、按置信度加权而非删除/软化，见 [07-next §13.1](07-next.md#131-方向-c难样本加权不删待启动)）。
