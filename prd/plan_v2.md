# NLPrompt Contest — 提分规划 V2（48% → 目标 60-75%）

> 更新时间：2026-08-07
> 背景：当前阶段 1-4 完成，线上准确率 **48%**（冻结骨干线性头）。
> 已落地方案 V2（LoRA 微调 CLIP ViT-B/32 骨干 + 动态筛选），正式训练进行中，含噪验证集 epoch4 已达 **81.5%**（远超 48%，印证骨干微调是主提分点）。
> 骨干均为"弱骨干"（CLIP ViT-B/32 / DINOv2 小模型），用户预期 75% 已算很高。

---

## 1. 差距根因分析

| 维度 | 本项目现状 (48%) | PGDF (59%) | 差距来源 |
|------|------------------|------------|----------|
| 骨干 | CLIP ViT-B/32 **冻结**，仅 512 维特征 | DINOv2 + **LoRA 微调骨干** | 骨干是否随任务进化（最大头） |
| 分类头 | 线性 cosine 头 + bias | LoRA 适配器 + 分类头 | 表达力 |
| 噪声处理 | 一次性 K-means 去噪 + 1 轮自训练扩展 mask | **动态 loss 筛选**：warmup 后每 5 epoch 按 CE loss 类内排名保留 top-r，叠加 prototype gate 取交集，迭代重训 | 净化强度与闭环 |
| 输入尺寸 | 224 | 448 | 分辨率带来的细节 |
| TTA | 翻转 | 多裁剪 + 翻转 | 推理鲁棒性 |

**关键认知（重要）**：
- 比赛 `requirements.md` 第 7-8 行明确定义任务为"在标签含噪条件下，对 **CLIP ViT-B/32 进行鲁棒微调**"。
- 约束"CLIP 权重固定"= 不得替换/重新预训练骨干权重，**但允许在其上做 PEFT/LoRA/全微调**（需求文档"技术方向"也明确列出 PEFT）。
- 因此本项目当前"冻结骨干只训线性头"是**过度保守**，把最容易拿分的部分放弃了。这是 48%→59% 差距的主因。

---

## 2. 提分路线图（按性价比排序）

### 路线 A：微调 CLIP 骨干（最关键，预期 +5~8 点）
不冻结 visual encoder，改为在 CLIP ViT-B/32 上加 **LoRA**（或 Adapter），用干净样本端到端训练。

- 借鉴 PGDF 的 **LORA + 分类头** 结构：
  - LORA 作用于 visual transformer 的 attention（`q/k/v/proj`），rank 视显存（MIG 2g.35gb）取 8~16，alpha=16/32。
  - 分类头 = `nn.Linear(512, 500)`（CLIP 原生 logit_scale 也放开）。
- 训练信号：用阶段 1 的 clean_mask 作为可靠监督；GCE 仍用于全量样本防噪声。
- 复用已有：`class_prototype.py` 的 clean_mask + features 可直接喂入新训练脚本。
- 实现：新建 `train_finetune.py`，从 `contest_clip.load_clip_to_cpu` 取 `clip_model.visual`，包一个 `LoRAVisualCLIP` 前向出 `(B,500)` logits。
- 推理：`test_prototype.py` 改写为从微调后 checkpoint 直接编码 + 分类头（不再用冻结特征 + 线性头）。

### 路线 B：迭代动态样本筛选（预期 +3~5 点）
把 PGDF 的"边训边筛"搬过来，套在路线 A 之上：

- warmup 5 epoch（用 clean_mask 监督）。
- 之后每 5 epoch：对**全部**训练样本算 CE loss，类内排序，每类保留 loss 最小 top-r（r=0.8）。
- 叠加 prototype gate：样本与阶段 1 原型余弦相似度需进每类 top-p（p=0.5）。
- 取 `loss_top_r ∩ proto_top_p` 作为下一轮训练集，迭代重训（约 3 轮）。
- 注意：比赛要求"噪声筛选必须可复现"——本方案纯自动、确定性（固定 seed），满足约束 4/5。

### 路线 C：更高输入分辨率（预期 +1~2 点）
- CLIP ViT-B/32 原生 224；PGDF 用 448。
- 可尝试 336（Vit-B/32 非原生但可插值位置编码，CLIP 支持 `image_encoder` 直接吃任意尺寸），或保持 224 但用更强的 TTA 补偿。

### 路线 D：噪声鲁棒损失组合（巩固）
- GCE(q=0.7) + 高置信度 CE 已用，沿用；新增 **loss 重加权**：对低置信度样本降权而非直接丢弃，保留更多边界样本。

### 路线 E：推理增强（低风险 +1~2 点）
- TTA 从单翻转 → 多尺度 + 多裁剪 + 翻转，soft-voting 多 checkpoint（`best.pt` + `ema_final.pt` + 末轮）。
- 注意比赛禁止多**模型**集成；同一模型多 checkpoint soft-vote 属单一模型，合规。

---

## 3. 优先级与执行顺序

1. **先做路线 A**（微调骨干）：单点收益最大，且是 48%→59% 的主要缺口。
2. 叠路线 B（动态筛选）：在微调基础上进一步净化，二者正交可叠加。
3. 叠路线 E（TTA/soft-vote）：推理期改动，零重训风险。
4. 视显存/时间再上路线 C（分辨率）。
5. 路线 D 作为损失细节贯穿全程。

目标分解：
- 仅 A：~53-56%
- A+B：~56-60%
- A+B+E：~58-63%
- 全上（含 C/D）：冲击 65-75% 区间（弱骨干上限）

---

## 4. 与现有产物的关系

- 阶段 1 `output/contest_prototype/*`：clean_mask / prototypes / features 全部复用，作为路线 A/B 的初始化与 gate。
- 阶段 2/3 旧 checkpoint（线性头）**不再作为最终提交**，但可作为路线 A 的 warm-start 参考或 ensemble 候选。
- 新产物建议目录：`output/contest_ft_lora/`（路线A）、`output/contest_ft_dynamic/`（A+B）。

---

## 5. 风险与约束自查（对照 requirements.md）

- 骨干=CLIP ViT-B/32 ✅（不替换，仅 LoRA 微调，权重文件固定）
- 无外部数据 ✅
- 无多模型集成 ✅（同模型多 ckpt soft-vote 合规）
- 可复现 ✅（固定 seed，自动筛选）
- 人工清洗非必要 ✅（全部自动）
- 提交格式 ✅（仍走 `test_prototype.py` 改版，输出 4 位零填充无表头 csv）

---

## 6. 下一步动作

1. 写 `train_finetune.py`（LoRA + 分类头，吃 clean_mask + GCE），后台训练。
2. 写 `selftrain_dynamic.py`（实现 loss_top_r ∩ proto_top_p 迭代筛选）。
3. 改写 `test_prototype.py` 支持从微调 checkpoint 推理 + 多 ckpt soft-vote + 多裁剪 TTA。
4. 跑通后用 `output/contest_ft_*/pred_results.zip` 提交，对比线上准确率。
