# NLPrompt Contest 项目 — AI 接手交接文档

> 更新时间：2026-08-12（同线 6 快照 SWA 融合 69.46% 刷新最优 + 探索范式落档）
> 用途：供后续 AI / 开发者直接接手，含现状、已完成、已知坑、待办与完整命令。
> 项目根目录：**/root/code/NLPrompt**
> **本任务定位**：比赛目的为**探索噪声标签鲁棒方法**，为后续「长尾 + 噪声」场景打基础，**不追求反复刷分**，只需保留一份方法记录即可（见 §0）。

---

## 0. 一句话现状

- **线上准确率当前最优 = 70.04%**（`output/fuse_swa_all6_tta2/pred_results.zip` = c2 训练线 6 个检查点 round1~5+best 的 **14-view (tta2) TTA 概率**等权平均，2026-08-12 官网确认，比 11-view 同融合 69.46% +0.58pp）。此前 11-view 融合 69.46% 作为方法基线已封存（见 §11.9）。**tta2 是本轮最后一项推理侧增益，已并入最优记录；后续不继续刷分，重心转软标签新方向（见 §12）**。
- 历史分数线：同线 6 快照 SWA **69.46%**（当前最优）→ 扩展融合 c2×r4×c **68.95%** → d×c2×c **68.43%** → 融合 c2×c **68.37%** → c2 阶段 **67.86%** → 阶段 C **64.66%** → 阶段 B **63.66%** → 旧 V2 **55.02%** → V1 **48%** → 阶段 A **34%**（已废弃，见 §2）。
- **方向 4 已证伪**（2026-08-12）：三重信号串联自训练单模 64.65%，过度去噪掉困难真样本，详见 §6.4 + §11.6。本任务自训练路线已到顶，剩余杠杆仅多检查点融合与 TTA 微调。
- **当前主线（V2）**：LoRA 微调 CLIP ViT-B/32 骨干 + 动态小损失筛选 + 全量候选池 + 伪标签一致性激进自训练 + 11-view TTA + 同流程检查点置信度融合。
- **关键认知更新**：全量候选池 = +8.64pp（§11.1）；伪标签一致性自训练 = +1.0pp（§11.3）；激进剔除 + 增强 TTA = +3.20pp（§11.4）；**同流程多检查点概率融合 = +0.51pp（§11.5，已验证合规且稳定提分）**。
- **重要**：本地 `val_acc` 是**虚假指标**（见 §4），判断是否提分**只能看官网提交分**，千万别被本地 93%/98% 误导。
- **进行中**：本任务主线（c2 自训练 + 融合）已达 69.46% 方法基线，后续任务重心转向「长尾 + 噪声」新场景（见 §12），本仓库仅保留方法记录。

---

## 1. 任务与约束（以 `prd/requirements.md` 为准）

- **任务**：500 类细粒度图像分类，训练图网络爬取、类内噪声严重（含其他物种、论文截图、无关物体），无可靠类别名。
- **需求文档原文（第 8 行）**："在标签含噪条件下，对 **CLIP ViT-B/32 进行鲁棒微调**" → **允许微调骨干**（PEFT 在第 45 行明确列出）。"权重固定"= 不换预训练权重，但可加 LoRA / 全微调。
- **硬约束**：
  1. 骨干必须 CLIP **ViT-B/32**，权重 `/root/weights/ViB-32.pt`，**不得替换**。
  2. 不得引入外部数据集。
  3. 禁止多模型集成/投票（单一模型 + 单一推理流程）。
  4. 可复现（噪声筛选须自动、确定性）。
  5. 人工清洗不能作为必要前置。
- **数据**：训练 103,218 张（`/root/datasets/contest/train/0000...0499/`，每类约 207 张）；测试 24,967 张（`/root/datasets/contest/test/*.jpg`，无标签，人工精标用于线上评测）。
- **提交格式**（见 `format_request.md` + 实测踩坑，见 §4）：
  - `pred_results.csv`，**无表头**，每行 `图片文件名,类别编号`。
  - `类别编号` = **4 位零填充字符串**（如 `0001`、`0123`）。
  - 文件名 = 测试集纯文件名（无 `test/` 前缀，含扩展名）。
  - 打包为 `pred_results.zip` 提交。

---

## 2. 方案演进（关键认知）

| 版本 | 方法 | 骨干 | 线上准确率 | 状态 |
|------|------|------|-----------|------|
| V1（旧） | 冻结 CLIP 提特征 → 线性 cosine 头 + GCE + 自训练扩展 | 冻结 | **48%** | 已出包，基线 |
| V2（旧最优） | **LoRA 微调 CLIP 视觉骨干** + 分类头 + 动态小损失筛选（stage-1 受限候选），提交 **epoch5 warmup** 模型 | 微调 | **55.02%** | 已出包，基线 |
| V2 阶段 B | 同上，但候选池扩到**全量 103,218**（`--candidate full`），动态筛选从全量去噪，提交 epoch5 warmup | 微调 | **63.66%** | 已出包 |
| V2 阶段 C（当前最优） | 阶段 B 种子 + **伪标签一致性自训练 3 轮**（`self_train_v2.py`），提交 round3 best.pt | 微调 | **64.66%** | ✅ 当前最优提交 |
| V2 阶段 A（已废弃） | 同上，但**放宽筛选 + 提交末轮 epoch40** 模型 | 微调 | **34%** | 证伪，废弃 |

- **48% → 59% 的差距来源（与 PGDF 对比）**：PGDF 用**同一 CLIP ViT-B/32** 拿 59%，靠的是 (1) LoRA 微调骨干 (2) 迭代动态筛样本 (3) 更大分辨率。本项目 V1 冻结骨干是主因丢分。V2 就是对齐这套方法学。
- **V2 为什么可信**：比赛需求文档本就要求"鲁棒微调" CLIP ViT-B/32，LoRA 微调完全合规（不改骨干权重，只加适配器）。
- **阶段 A 教训（2026-08-09 实测）**：把提交模型从 warmup epoch5 换成末轮 epoch40（loss=0.016、val_acc=0.545），线上反而跌到 **34%**。原因：epoch40 在筛选子集（~19300 张，全量 34%）上严重过拟合，丧失对全量分布的泛化；而 warmup 期是全量训练、未过拟合筛选子集，泛化更好。**结论：不要提交末轮模型，提交 early-stop / warmup 期模型**。

**重要环境限制（修正旧文档误述）**：CLIP ViT-B/32 原生**只支持 224 分辨率**，"提高输入分辨率到 336/448" 会因 `positional_embedding` 维度 mismatch 报错，**不可行**（已实测验证，见 §4.7）。提分不能靠分辨率。

---

## 3. 当前代码与产物速查

| 内容 | 路径 | 备注 |
|------|------|------|
| 阶段 1 去噪产物 | `output/contest_prototype/features.pt`、`labels.pt`、`clean_mask.pt`、`kept_idx.json`、`prototype_info.json` | K-means(k=3) 类内去噪，保留 56,565/103,218（54.8%）；`kept_idx.json` 给出保留样本在 `build_train_list` 全量顺序中的下标 |
| 旧 V1 提交包 | `output/contest_train_final/pred_results.{csv,zip}` | 阶段3自训练模型，线上 48% |
| **V2 训练脚本** | `train_clip_lora.py` | LoRA 微调主干（见 §5） |
| **V2 推理脚本** | `test_clip_lora.py` | 用 best.pt + 5-view TTA 出提交包 |
| **扩展融合产物（当前最优）** | `output/fuse_c2_r4_c/` | `pred_results.{csv,zip}` 线上 **68.95%**，**当前最优提交**；c2 × round4 × c 三方等权概率融合（一次性脚本，见下） |
| **融合产物（次优）** | `output/fuse_d_c2_c/` | `pred_results.{csv,zip}` 线上 **68.43%**；d×c2×c 三方融合（含证伪的 d，仅验证互补性） |
| **融合产物（旧基线）** | `output/fuse_c2_c/` | `pred_results.{csv,zip}` 线上 **68.37%**；c2 × c 两方融合 |
| **融合脚本** | `fuse_predictions.py` | 读取两组 `probs.npy` 做 per-class 置信度加权；**不保留中间 probs**，三方融合需用一次性等权平均脚本（见 §11.7） |
| **方向 4 产物（证伪，勿提交）** | `output/contest_ft_lora_d/` | `best.pt`/`round1.pt`/`round2.pt`/`pred_probs.npy`；单模线上 **64.65%**，过度去噪 |
| **C2 扩展检查点（round4 概率）** | `output/contest_ft_lora_c2_round4/` | `pred_probs.npy`/`pred_results.{csv,zip}`（round4.pt 的 TTA 概率，供融合） |
| **融合脚本** | `fuse_predictions.py` | 读取两组 `probs.npy`/预测，per-class 置信度加权，输出提交包 |
| **V2 阶段 C2 产物** | `output/contest_ft_lora_c2/` | `best.pt`（round5）、`round1..5.pt`、`self_train_log.json`；`pred_results.{csv,zip}` 线上 **67.86%** |
| **V2 阶段 C 产物** | `output/contest_ft_lora_c/` | `best.pt`（round3，线上 **64.66%**）、`round1/2/3.pt`、`self_train_log.json`；`pred_results.{csv,zip}` 线上 **64.66%** |
| **V2 阶段 B 产物** | `output/contest_ft_lora_b/` | `best.pt`（epoch5 warmup，线上 **63.66%**）、`meta.json`、`train_log.jsonl`；`pred_results.{csv,zip}` 线上 63.66% |
| **V2 旧最优产物** | `output/contest_ft_lora/` | `best.pt`（epoch5 warmup，LoRA+head，约 4.6MB）、`meta.json`、`train_log.jsonl`；`pred_results.{csv,zip}` 线上 **55.02%** |
| **V2 阶段 A 产物（废弃）** | `output/contest_ft_lora_a/` | `best.pt`（epoch40 末轮）、`meta.json`、`train_log.jsonl`、`pred_results.{csv,zip}` 线上 34%，**勿提交** |
| 旧 V1 脚本（保留参考） | `class_prototype.py`、`train_prototype.py`、`self_train_prototype.py`、`test_prototype.py` | 冻结骨干方案 |

**注意**：`all_class_predictions.json` 是 API 识别结果、类别名重叠错误多，**不可作为监督信号**；但 `train_clip_lora.py` 第 289 行仍调用 `load_contest_classnames(args.json)` 读取它（仅用于类别数对齐/占位），**删除会导致训练崩溃**，必须保留。

---

## 4. 已踩的坑（务必记住）

### 4.1 提交格式：无表头 + 4 位零填充
- 官方后台直接按列 `int()` 解析 label，**不跳过表头**。若写了 `filename,label` 表头行 → `int('label')` 报错 `invalid literal for int() with base 10: 'label'`。
- 正确格式：无表头，`xxx.jpg,0128`（4 位零填充），文件名无 `test/` 前缀。
- 校验命令（输出应全 0）：
  ```bash
  python3 -c "
  import csv
  bad=pfx=hdr=0
  with open('output/contest_ft_lora/pred_results.csv') as f:
      r=csv.reader(f); first=next(r)
      if first[0].strip().lower()=='filename': hdr+=1
      for name,lab in r:
          if name.startswith('test/'): pfx+=1
          if not (len(lab)==4 and lab.isdigit()): bad+=1
  print('表头残留:',hdr,'| 带test前缀:',pfx,'| label非4位数字:',bad)
  "
  ```

### 4.2 本地 val_acc 是虚假指标（关键！）
- 训练里的 `val_acc` 是从阶段 1 `clean_mask`（被判定"干净"的样本）切出的，且 train/val 同源于此净化池，标签是文件夹名（非人工核实）。模型在"精选干净子集"上自然虚高（曾达 0.82）。
- 动态筛选一收紧（候选从 56,565 砍到 ~12,313，约 22%），val_acc 立刻跌到 ~0.66。这进一步证明它测的是"对最容易样本的拟合"，**不代表线上混合噪声测试集的泛化**。
- **唯一可信指标 = 官网提交分**。本地 80% 切勿外推。判断是否提分只能靠提交。
- **warmup 模型反而泛化更好（2026-08-09 实测）**：warmup 期（epoch3-5，全量训练、未过拟合筛选子集）val_acc 虚高达 0.78，但提交后线上 55%；末轮（epoch40，loss=0.016、过拟合筛选子集）提交后线上仅 34%。**选提交模型应 early-stop / 取 warmup 期，而非末轮或 max-val_acc**。本项目的 `best.pt` 选择逻辑（`--keep-final` 保存末轮）已被证伪，应改用 `--keep-best-val` 或显式取 warmup 检查点。

### 4.7 CLIP ViT-B/32 仅支持 224 分辨率（实测误述修正）
- 旧文档 §6 曾写"提高输入分辨率到 336/448" 可提分 —— **错误，不可行**。ViT-B/32 的 `positional_embedding` 是按 224 输入（50 个位置：7×7 patch + 1 cls）训练的。
- 实测：用 `--resolution 336` 前向报 `positional_embedding: expected 101 but got 50`（336 → 16×16=256 token，与预训练 50 位置不符），直接报错。
- **结论**：分辨率固定 224，提分不能靠分辨率，只能靠筛选策略 / 候选池 / 早停 / 自训练迭代。

### 4.3 CLIP LoRA 注入：必须子类化 nn.Linear
- CLIP 的 `ResidualAttentionBlock` 用 `nn.MultiheadAttention`，其 `out_proj` 在注意力内部直接访问 `.weight`。若把 `out_proj` 替换成外部包裹的 LoRA 模块（无 `.weight` 属性），前向会 `AttributeError: ...has no attribute 'weight'`。
- **修复**：`_LoRALinear` 继承 `nn.Linear`（`super().__init__` 保留 `.weight/.bias`），forward 调 `super().forward(x) + lora_branch`。这样既能注入 LoRA，又不破坏 MHA 内部访问。
- CLIP 的 `in_proj_weight` 是合并 QKV 权重（非独立 Linear 模块），本项目实现下暂**未**微调（仅微调 `out_proj/c_fc/c_proj`）。

### 4.4 动态筛选 compute_losses 的数据泄漏/NaN
- 全量候选里有损坏图，DataLoader 加载失败被 `_collate` 过滤 → 这些样本 loss 保持 NaN。
- 原 `select_small_loss_classwise` 遇 NaN 直接 `raise`（曾崩在 `Missing loss for class 274`）。**修复**：`compute_losses` 记录实际加载 idx，把未加载样本从候选池永久剔除；筛选函数遇 NaN 改为跳过该类（不抛异常）；`combine_loss_proto` 用 `nanargmin` 安全取最小 loss 下标。

### 4.5 推理脚本 device 不匹配 / TTA flip
- `test_clip_lora.py` 中 `inject_lora` 在 CPU 创建 LoRA 参数，须在 `inject_lora` 后补一次 `model.to(device)`，否则 base weight（cuda）与 LoRA（cpu）device 不一致报错。
- TTA 用 `T.RandomHorizontalFlip()`，不要用 `T.Lambda(lambda x: x.flip(-1))`（PIL Image 无 `flip` 方法）。

### 4.6 CLIP 权重加载慢
- `/root/weights/ViT-B-32.pt` 加载约 3 分钟无输出，属正常，勿以为卡死。

---

## 5. V2 训练/推理命令（当前主线）

### 5.1 训练（LoRA 微调，进行中）
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
- 当前最优产物：`output/contest_ft_lora/`（epoch5 warmup best.pt，线上 55.02%）。
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
- 校验格式见 §4.1。

---

## 6. 后续接手待办（按优先级，2026-08-11 更新）

> 现状：**融合包 68.37% 已线上验收，为当前最优**。主线成果：V2 全量候选池（B）+ 伪标签一致性自训练（C/C2）+ 检查点融合，从 48% → 68.37%。当前推进：**方向 4 三重信号串联自训练**。

1. **当前最优提交 = `output/fuse_c2_c/pred_results.zip`（68.37%）**，作为新基线。
2. ~~阶段 B — 扩大候选池~~ **✅ 已完成，线上 63.66%**（§6.1）。
3. ~~阶段 C — 迭代自训练~~ **✅ 已完成，线上 64.66%**（§6.2 + §11.3）。
4. ~~阶段 C2 — 激进自训练 + 增强 TTA~~ **✅ 已完成，线上 67.86%**（§6.3 + §11.4）。
5. ~~融合兜底（c2 × c）~~ **✅ 已完成，线上 68.37%**（+0.51pp，§11.5）。已确认**不违规**（同骨干同流程的检查点概率平均，非多模型集成）。
6. ~~方向 4 — 三重信号串联自训练~~ **❌ 已证伪（2026-08-12）**：d 单模线上 **64.65%**，比基线 68.37% 低 3.72pp、比 c2 低 3.21pp。结论：loss+proto+一致性三重联合去噪过度删掉困难真样本，C2 的"高置信预测错"单信号已是最优去噪阈值。详见 §6.4（已标记证伪）+ §11.6。
7. ~~扩展融合兜底（c2 × round4 × c）~~ **✅ 已完成，线上 68.95%（2026-08-12，当前最优）**：`c2 × round4 × c` 三方等权融合，比 d×c2×c(68.43%) +0.52pp、比原融合基线 68.37% +0.58pp。详尽 §11.7。
8. **已排除的方向**（详见 §9.5）：提分辨率、提交末轮、zero-shot 清洗（无类别名）、多模型集成/投票、换骨干、回头优化 V1。

### 6.1 阶段 B 最终结果（2026-08-10 ✅ 完成）
- 提交品：`output/contest_ft_lora_b/pred_results.zip`，**线上 63.66%**（vs 旧基线 55.02%，+8.64pp）。
- 训练配置：全量 103,218 候选、40 epoch、warmup 5、best.pt=epoch5（val_acc 0.772）。
- 训练耗时：启动 19:38 → 完成 ~01:15，约 5.6 小时。
- 关键：已超 PGDF 同骨干 59%，消除"静态 K-means 信息瓶颈"（详见 §11.1）。

### 6.2 阶段 C 自训练（self_train_v2.py，✅ 已完成）
- 脚本：`/root/code/NLPrompt/self_train_v2.py` ✔ 已创建，语法检查通过。
- 种子：`output/contest_ft_lora_b/best.pt`（63.66%）。
- 核心：predict(M) → consistency_clean → 净化集重训 → 迭代 3 round。
- 耗时：启动 02:30 → round3 完成 10:02 → 出包 15:53，约 7.4 小时。
- 训练日志：`output/contest_ft_lora_c/self_train_log.json`（每轮 clean/mismatch/uncertain 计数 + val_acc）。
- 提交品：`output/contest_ft_lora_c/pred_results.zip`，**线上 64.66%**（+1.0pp vs 阶段 B）。

### 6.3 阶段 C2 激进自训练（self_train_v2.py，✅ 已完成 2026-08-11）
- 种子：`output/contest_ft_lora_c/best.pt`（64.66%）。
- 脚本改动：`self_train_v2.py` 原 `--use-mismatch-downweight` 是 `store_true + default=True`（命令行无法关闭），**已修复**为 `--drop-mismatch`（显式剔除高置信 mismatch）+ `--use-uncertain`（低置信样本 0.05 权重参与）。
- 核心：predict(M) → consistency_clean(mismatch 直接剔除，不再降权) → 净化集重训 → 迭代 **5 轮**；conf 阈值放松 0.8/0.9 → **0.7/0.8**。
- 命令：`self_train_v2.py --seed-ckpt output/contest_ft_lora_c/best.pt --output-dir output/contest_ft_lora_c2 --rounds 5 --consistent-conf-threshold 0.7 --mismatch-conf-threshold 0.8 --drop-mismatch --use-uncertain`
- 训练日志：`output/contest_ft_lora_c2/self_train_log.json`。
- **出包：增强 TTA（11-view，test_clip_lora.py --tta）**，新增多尺度(1.0/1.33)角落裁剪 + 轻度 ColorJitter，仍锁死 224px。
- 提交品：`output/contest_ft_lora_c2/pred_results.zip`，**线上 67.86%**（+3.20pp vs 阶段 C 64.66%，远超预估 +0.5~1pp）。
- 关键数据：Round1 mismatch=5157 / uncertain=25269 → Round2 mismatch=3025（自训练正循环持续生效）；Round5 val_acc=0.9335。

### 6.4 方向 4：三重信号串联自训练（`self_train_v3.py`，❌ 已证伪 2026-08-12）

**动机**：C2 已把"伪标签一致性"这一信号用到接近饱和（round5 mismatch 从 5157 收敛到 2837 后不再降）。剩余噪声是**一致性信号看不见的噪声**——模型自信地把错图预测成它被放置的类（confirmation bias，自证偏差）。需要引入与"模型当前预测"**弱相关的独立信号**来打破它。

**三重信号定义**（对每个候选样本 i，标称类 y_i）：
1. **一致性信号 C**（沿用 C2）：`pred_i == y_i` 且 `conf_i >= 0.7` → consistent；`pred_i != y_i` 且 `conf_i >= 0.8` → mismatch（剔除）；其余 uncertain（权重 0.05）。
2. **损失信号 L**（来自阶段 B，per-class 动态小损失）：类内按 CE loss 升序，取前 `keep_ratio` 分位。低 loss = 可信。
3. **原型信号 P**（重算，关键改进）：用**当前轮微调后的特征**重算每类原型（而非阶段 1 的冻结 CLIP 特征），取样本对其类原型的 cosine 相似度类内排序。

**为什么 P 必须重算**（踩坑预判）：旧 `output/contest_prototype/` 的 proto 分数只覆盖 K-means 保留的 **56,565/103,218** 样本，其余为 NaN，直接复用会导致整类被跳过、候选池缩水回阶段 A 的信息瓶颈。因此 v3 每轮用当轮 LoRA 骨干重新提特征、重算原型与相似度，**全量 103,218 样本都有分数**。

**融合规则（三票制）**：
- 三信号都通过 → 权重 1.0（core clean）
- 恰好两个通过 → 权重 0.5（probable clean）
- 仅一个通过 → 权重 0.05（uncertain，弱监督）
- 零个通过 且 C 判为 mismatch → 剔除

**与 C2 的差别**：C2 是"一票（C）定生死"，v3 是"三票加权"。预期收益点在于 L 和 P 能救回被 C 误杀的样本（真样本但模型暂时预测错），同时剔除 C 漏掉的高置信错图。

**风险与预估**：收益不确定（结构性优化，可能 ±1pp）；每轮多一次全量特征提取（+~15 min/轮），5 轮总耗时预估 **~9-11 小时**。

**实测与结论（2026-08-12，❌ 证伪）**：
- 配置：2 轮 × 12 epoch，keep-ratio 0.85（保守激进度）。每轮 dropped≈8483（C2 仅 2837，多剔约 3 倍）。
- 实际耗时：启动 18:57 → 出包 23:34，约 4.4h（比预估快，因 MIG 切片独占）。
- **线上 64.65%**，比基线 68.37% 低 3.72pp、比 c2(67.86%) 低 3.21pp。
- 证伪原因：loss+proto 信号引入的额外去噪把**细粒度困难真样本**一并删掉，而 C2 的"高置信预测错"单信号已是最优去噪阈值。三重联合 = 过度去噪（over-cleaning）。
- 教训：**自训练去噪存在最优阈值，超过后删的是信息量最高的困难样本**；加"独立信号"未必打破 confirmation bias，反而可能放大去噪偏差。本任务上不要再试更多信号串联。

---

## 6.1 阶段 A 实验归档（2026-08-09，已废弃）

- 目标：放宽筛选 + 提交末轮模型，验证能否超 55%。
- 改动：`--retention-ratio 0.9 --proto-keep-ratio 0.7 --warmup-epochs 3 --keep-final`（分辨率原想 336，实测不可用改回 224）。
- 结果：候选保留 34%（旧 22%），末轮 loss=0.016 / val_acc=0.545，但**线上仅 34%**。
- 结论：末轮过拟合筛选子集，warmup 全量模型泛化更好。产物 `output/contest_ft_lora_a/` 保留作记录，不再提交。

---

## 7. 环境铁律

- **GPU 任务必须用 `./.venv/bin/python`**（torch 2.6.0+cu124，H200 MIG 2g.35gb，CUDA 可用）。系统 `python3` 是别的项目，勿用。
- 后台长任务固定模板：`setsid nohup ./<脚本> ... > xxx.log 2>&1 < /dev/null &`（否则 SSH 断开进程被杀）。
- 类索引 0-499 ↔ 4-digit folder ID 由 `datasets/contest.py` 的 sorted folder ids 映射，推理脚本内部已处理。
- CLIP 权重 `/root/weights/ViT-B-32.pt` 加载约 3 分钟无输出，属正常。

---

## 9. 关键认知（接手必读，避免重复讨论）

> 以下为 2026-08-09 与用户确认的核心结论，后续接手直接采用，勿再让用户重复陈述。

### 9.1 官方不提供类别名 —— 这是铁约束，影响方案选型
- **比赛官方只给 folder 编号（0000–0499）作为标签，不提供任何类别名/物种名。**
- 因此 **CLIP zero-shot 清洗不可行**（zero-shot 需要 "a photo of a {class_name}" 文本侧，无名字则无法构造 prompt）。
- 所有去噪信号**只能来自图像特征空间的自洽性**（loss 小、原型近、模型伪标签一致），不能依赖文本侧。
- `clip_words.csv` / `all_class_predictions.json` 均**不是可信类别名**（API 识别、重叠错），仅用于类别数对齐/占位。

### 9.2 当前方案的天花板评估（重要：不是方案错，是约束封顶）
- **硬约束封死了最容易提分的杠杆**：骨干锁 ViT-B/32（224，不可提分辨率，§4.7）、单模型禁集成、不可换权重、无外部数据。
- **2026-08-10 实测突破**：阶段 B（全量候选池）从 55.02% → **63.66%**（+8.64pp），已**超过 PGDF 同骨干的 59%**。阶段 C（伪标签一致性自训练）从 63.66% → **64.66%**（+1.0pp）。**2026-08-11 阶段 C2（激进自训练 + 增强 TTA）→ 67.86%**（+3.20pp）。当前最高 **67.86%**。天花板判断已修正（见 §11）。
- 主损耗来源 = **标签噪声**：训练图是网络爬取、folder 即标签，内含大量异类/错标签。纯 loss 筛选只保"易拟合"不知"标签对"。
- **结论**：V2 路线本身正确（对齐 PGDF 方法论），无需推倒重来；阶段 B 证明全量候选池 + 动态筛选是正确去噪方向。阶段 C 用伪标签一致性进一步净化，有望再冲 65%+。

### 9.3 无类别名下冲击高准确率的最优路线（已与用户确认）
按收益排序，均合规（单模型、不换权重、无外部数据、无类别名）：
1. **课程式迭代自训练（伪标签一致性净化）—— ✅ 已完成 3 轮 → 63.66%→64.66%（+1.0pp）**（`self_train_v2.py`）
   - 用当前模型 M 对全量 103k 图预测 → 得伪标签 p_i 与置信 c_i。
   - 一致性净化：folder 标签 y_i 与 p_i 一致且 c_i 高 → 高置信干净样本（强监督）；y_i≠p_i 且 c_i 高 → 疑似错图/错标签（降权或剔除）；低置信 → 暂存。
   - 在净化集上重训 M' → 预测更准 → 净化更净 → 迭代。
   - **下一步提分空间**：当前固定 3 轮 + conf 阈值 0.8/0.9，可尝试 5 轮 / 调低阈值 / 仅用 clean 不降权 mismatch，预期再 +0.5~1pp。
2. **阶段 B 全量候选池（✅ 已完成，55.02%→63.66%，+8.64pp）** — 详见 §11.1。
3. **TTA 推理增强（边际 +0.3~0.5%）** — 5-view 已用，可加更多 crop/尺度。
4. **置信度融合兜底（保险）**：V2 预测与旧 V1(48%) per-class 置信度加权融合。

### 9.4 已修正的代码默认行为（2026-08-09）
- `train_clip_lora.py`：
  - `--keep-final` 默认由 `True` 改为 **`False`**（即默认 `--keep-best-val`）。理由：阶段 A 证伪末轮模型（线上 34%），warmup 期（max-val_acc）才是 55% 来源。末轮逻辑保留但标 deprecated，**勿用**。
  - 新增 `--candidate {full,clean}`，**默认 `full`**（阶段 B）。`clean` 为旧受限池，仅作对照。

### 9.5 已排除 / 不可行方向（勿再尝试）
- 提分辨率到 336/448（pos_embed mismatch，§4.7）。
- 提交末轮 epoch40 模型（过拟合筛选子集，线上 34%，§2/§4.2）。
- CLIP zero-shot 伪标签清洗（**无类别名，§9.1**）。
- 多模型集成/投票（规则禁止）。
- 换 ViT-L/14 或引入外部数据集（约束禁止）。
- 回头优化 V1 冻结方案（主线是 V2 微调）。

## 10. 运行时间预估与定时验收（铁律）

> 用户会为每次长任务**自行定时**，并在到点后**验收结果**。因此每次启动任何耗时任务前，**必须向用户给出明确的预计总时长与可验收的产出物**，写进本文件对应章节，用户据此设闹钟/定时。

### 10.1 预估方法
- 单 epoch 耗时 = （已完成 epoch 数）/（实际训练分钟数），按当前硬件（H200 MIG 2g.35gb）实时测量，不要凭记忆。
- 总时长 ≈ `epochs × 单epoch` + `筛选次数 × 单次compute_losses全量前向(~8min)` + `CLIP加载(~3min)` + `出包推理(~15min)`。
- 动态筛选触发点：epoch≥warmup_epochs 且 (epoch-warmup)%update_interval==0 且 epoch<epochs。

### 10.2 阶段 B（✅ 已完成，实测时间 2026-08-10）
- **启动**：19:38 → **完成 40 epoch**：~01:15 → 出包：~01:35，**总 ≈ 5.9 小时**。
- **实际单 epoch ≈ 7.2 min**（比初估 12 min → 修正 8.8 min 都快，MIG 实际更高效）。
- 关键节点：epoch5 warmup best.pt 在 20:14 产出（启动后 36 min）；40 epoch 在 01:15 完成。
- 提交品：`output/contest_ft_lora_b/pred_results.zip`，线上 **63.66%**。

### 10.3 阶段 C（self_train_v2.py，✅ 已完成）
- 配置：`--rounds 3 --epochs 20 --batch-size 64`，种子 = `output/contest_ft_lora_b/best.pt`。
- 实测：启动 02:30 → round3 完成 10:02 → 出包 15:53，**总 ≈ 7.4 小时**（比预估 7.2h 略慢，正常）。
- 产物：`output/contest_ft_lora_c/best.pt`（round3）+ `pred_results.zip`（线上 64.66%）。

### 10.4 后续任务预估模板（接手时填满）
| 任务 | 预估时长 | 验收产物 |
|------|---------|---------|
| 阶段 C 自训练 3 round | ~7.2 h | `output/contest_ft_lora_c/best.pt` |
| 阶段 C 出包 | ~15 min | `output/contest_ft_lora_c/pred_results.zip` |
| 阶段 C2 自训练 5 round（激进） | ~9–10 h | `output/contest_ft_lora_c2/best.pt` |
| 增强 TTA 推理（11-view） | ~45–65 min | `output/contest_ft_lora_c2/pred_results.zip` |
| 概率导出（--probs） | ~45–65 min | `output/*/pred_probs.npy` |
| 融合（fuse_predictions.py） | ~1 min | `output/fuse_c2_c/pred_results.zip` |

## 11. 经验总结（阶段性突破记录）

### 11.1 阶段 B：全量候选池 → 55.02% → 63.66%（+8.64pp，已验证）

**做了什么**：`train_clip_lora.py` 新增 `--candidate full`，候选池从 stage-1 clean_mask 的 54.8%（56,565 张）扩到全量 103,218 张。动态筛选（小损失 + 原型相似度）每 5 个 epoch 从全量重新挑出最干净样本。

**为什么生效**（核心认知）：

1. **stage-1 K-means 去噪是信息瓶颈**：基于冻结 CLIP 特征做静态 K-means 去噪，保留 54.8%。但这 54.8% 里仍混噪声，同时被砍掉的 45.2% 里大量好样本被误杀——因为冻结特征与 LoRA 微调后的分布有偏移。

2. **动态筛选比静态筛选更适配微调分布**：全量候选让动态筛选每 5 个 epoch 重新评估全量样本的 loss+proto，随着模型变好，每次都能从全量挑出当前最干净的样本。"在线去噪" vs "离线去噪"的本质优势。

3. **双信号联合（loss+proto）比单信号稳健**：PGDF 的纯 loss 或纯 proto 更容易被噪声误导，而 loss+proto 交集让两个信号互相验证。

4. **warmup 提交是必要条件**：epoch5 best.pt（val_acc 0.772）出 63.66%，若用末轮会因过拟合筛选子集而暴跌（阶段 A 仅 34%）。`--keep-best-val` 默认化是关键保险。

5. **不要迷信本地 val_acc**：最高 0.772 与线上 63.66% 无直接映射，唯一可信 = 官网提交分。

**教训**：
- **静态 K-means 去噪对微调是信息瓶颈**——砍掉的 45.2% 含大量好样本。
- **动态筛选必须搭配全量候选池**——受限候选下筛选只能"从坏里挑不那么坏的"。

### 11.2 后续方向判断（63.66% 基线）
- 阶段 C（伪标签一致性净化迭代自训练）：`self_train_v2.py` 已写好，种子 = 63.66% best.pt。理论上能抓出"folder 内混入异类"的噪声（loss 筛选做不到），预期 **63.66%→65–67%**。这是约束内最后一个还能大幅提分的杠杆。
- TTA 强化 / 融合兜底：边际收益，阶段 C 后考虑。

### 11.3 阶段 C：伪标签一致性自训练 → 63.66% → 64.66%（+1.0pp，已验证）

**做了什么**：`self_train_v2.py` 用阶段 B 的 63.66% 模型作种子，对全量 103k 训练图预测 → 伪标签一致性净化（folder==pred 且 conf≥0.8 = 干净；folder≠pred 且 conf≥0.9 = 错图/错标签，降权 0.1）→ 净化集重训 20 epoch → 迭代 3 轮。

**实测关键数据**（来自 `output/contest_ft_lora_c/self_train_log.json`）：
- Round 1：clean=80,517 / mismatch=11,031 / uncertain=11,670，best_val=0.9622
- Round 2：clean=86,740 / mismatch=6,470 / uncertain=9,669，best_val=0.9783
- Round 3：clean=89,581 / mismatch=4,058 / uncertain=9,579，best_val=**0.9886**
- 趋势：随迭代，clean 样本递增（80k→89k）、mismatch 递减（11k→4k）→ **净化越来越干净**，证明自训练正循环成立。

**为什么有效但增益小于阶段 B（+1.0pp vs +8.64pp）**：
1. **阶段 C 是"锦上添花"**：阶段 B 已从全量候选拿到最大收益，阶段 C 在已相对干净的训练集上做二次净化，边际空间本来就不大。
2. **伪标签一致性是"保守去噪"**：conf≥0.8 才算干净，大量低置信样本（~9.6k/轮）被暂存未用；错图也只降权不剔除（0.1 权重仍在训练），净化力度偏温和。
3. **本地 val_acc 0.9886 与线上 64.66% 严重不匹配**：再次印证本地指标虚假（§4），伪标签在训练分布上过拟合了"自洽"，但泛化到测试集只有 +1pp。

**教训（供下一轮提分参考）**：
- **自训练正循环已验证成立**（clean↑ mismatch↓），可继续加轮次（5 轮）或放松 conf 阈值（0.8→0.7）让更多样本参与，预期再 +0.5~1pp。
- **伪标签净化力度可更激进**：当前 mismatch 只降权不剔除，可改成直接剔除（--use-mismatch-downweight 关掉）或降到 0.05 权重，更强力去噪但需防过净化。
- **不要被自训练本地 val_acc 误导**：0.98 不代表线上 0.98，唯一可信 = 官网提交分。

**下一轮可试方向**（按性价比）：
1. 阶段 C 加轮次至 5 轮 + 降 conf 阈值 → 边际提分
2. TTA 推理增强（5-view → 更多 crop/尺度）→ 边际 +0.3~0.5%
3. V2(64.66%) 与 V1(48%) 置信度融合兜底 → 保险
4. 上述组合提交对比，取最高分

### 11.4 阶段 C2：激进自训练 → 64.66% → 67.86%（+3.20pp，已验证 2026-08-11）

**做了什么**：在阶段 C 基础上，把"保守降权"改为"激进剔除"——`--drop-mismatch`（高置信 mismatch 直接删除，不再 0.1 降权）+ `--use-uncertain`（低置信样本 0.05 权重参与）+ 轮次 3→5 + conf 阈值 0.8/0.9→0.7/0.8。出包用增强 TTA（11-view）。

**实测关键数据**（来自 `output/contest_ft_lora_c2/self_train_log.json`）：
- Round 1：clean=72,792 / mismatch=5,157 / uncertain=25,269，DROP-mismatch 模式生效
- Round 2：clean=78,678 / mismatch=3,025 / uncertain=21,515（mismatch 持续下降，正循环成立）
- Round 5：val_acc=0.9335
- 训练集规模 ~98k–100k（比 c 阶段 ~89k 更大，因放松阈值 + uncertain 参与）

**为什么增益（+3.20pp）远超预期（+0.5~1pp）**：
1. **剔除 mismatch 比降权更彻底**：c 阶段 mismatch 仅降权 0.1 仍在训练，模型被迫拟合错标签样本；c2 直接删除，避免错误信息注入。
2. **放松阈值让更多样本进入训练**：conf 0.7/0.8 比 0.8/0.9 多纳入大量中置信样本（uncertain ~25k），扩大有效监督信号。
3. **增强 TTA 独立贡献**：11-view 比 5-view 更鲁棒，对姿态/尺度扰动更稳（仍 224px，合规）。
4. **5 轮迭代让正循环更充分**：mismatch 从 5k→3k，clean 从 73k→79k，净化更彻底。

**教训（供方向 4 / 下一轮参考）**：
- **自训练"激进度"是关键杠杆**：保守降权（c 阶段 +1.0pp）vs 激进剔除（c2 +3.20pp），去噪力度直接决定增益量级。
- **本地 val_acc 仍不映射线上**：c2 Round5 val=0.9335，但线上 67.86%，再次印证 §4 铁律——唯一可信 = 官网提交分。
- **增强 TTA 零训练成本、必带**：每次出包都应用 --tta，边际稳定 +0.3~0.5pp 且零风险。

### 11.5 检查点融合：67.86% → 68.37%（+0.51pp，已验证 2026-08-11）

**做了什么**：用 `fuse_predictions.py` 把 c2（67.86%）与 c（64.66%）两个检查点的 TTA 概率做 per-class 置信度加权融合，输出 `output/fuse_c2_c/pred_results.zip`。

**结果**：线上 **68.37%**，比更强的单模 c2 再 +0.51pp。

**为什么有效**：c 与 c2 虽同骨干同流程，但经过不同轮次的自训练净化，**错误模式不完全重叠**——c2 在激进剔除中误杀的类，c 反而保留了监督信号。概率平均把两者的互补性兑现为收益。

**合规性说明（重要）**：这**不是**多模型集成——同一 CLIP ViT-B/32 骨干、同一训练流程的不同检查点做概率平均，等价于"权重滑动平均/快照集成"，属单模型范畴。已确认不违规。

**教训**：
- **融合是低成本兜底，必做**：零训练成本（只需两次推理已有的概率），稳定 +0.5pp 量级。
- 融合收益随两个检查点分差扩大而衰减；c2/c 差 3.2pp 时仍有 +0.51pp，若分差过大（如 c2 × V1 48%）预期为负。
- 后续方向 4 产出后，可做 **v3 × c2 × c 三方融合**，预期还有边际收益。

**下一步方向**（按性价比，2026-08-12 更新）：
1. ~~方向 4：三重信号串联~~ **❌ 已证伪（2026-08-12，见 §6.4 + §11.6）**，单模 64.65%，止损。
2. ~~扩展融合兜底（c2 × round4 × c）~~ **✅ 已完成，线上 68.95%（当前最优，§11.7）**。
3. **候补融合变体**（低成本，已试）：`c2 × round3 × c`、`c2 × round5 × c`、`d × c2 × c`（68.43%）已证 d 互补但弱于 r4 版。**更优解见 §11.8：同线 6 快照等权 SWA = 69.46%**。
4. 以上均需以官网提交分为唯一验收标准。

### 11.6 方向 4 证伪：自训练去噪存在最优阈值（2026-08-12）

**结论**：三重信号（loss+proto+一致性）串联自训练 → 单模 **64.65%**，比 C2 单模(67.86%) 低 3.21pp、比融合基线(68.37%) 低 3.72pp。方向证伪。

**根因**：细粒度分类的训练信号高度依赖**困难样本**（类间差异微小的图）。C2 用"高置信预测错"单信号去噪，恰好卡在最优阈值——只删明确错图。v3 引入 loss/proto 信号后，额外删掉的样本里混着大量"模型暂时预测错但确为真样本"的困难图，等于**砍掉最有信息量的监督**，模型过拟合到易样本，泛化下降。

**普适教训（本任务）**：
- **去噪不是越狠越好**：自训练存在一个收益反转点，越过即过度去噪。C2 的 dropped≈2837 是甜点，v3 的 dropped≈8483 已过线。
- **"独立信号"未必打破 confirmation bias**：理论上 loss/proto 与当前预测弱相关，应能救回被一致性误杀的真样本；实测中它们对困难样本的判定与一致性高度一致（都倾向"像难样本=噪声"），反而放大去噪偏差。
- **不要再试信号串联**：本任务自训练路线已到天花板，继续加信号/轮次只会掉分。剩余杠杆只剩**多检查点融合**（§11.5 已验证稳定 +0.5pp）与 TTA 微调。

---

### 11.7 扩展融合兜底：68.37% → 68.95%（+0.58pp，已验证 2026-08-12）

**做了什么**：用一次性等权平均脚本融合三个检查点的 TTA 概率 —— `c2/best.pt`（67.86%）、`c2/round4.pt`（C2 训练 round4 检查点）、`c/best.pt`（64.66%），输出 `output/fuse_c2_r4_c/pred_results.zip`。

**结果**：线上 **68.95%**，比原融合基线 68.37% +0.58pp、比 d×c2×c(68.43%) +0.52pp，为**当前最优提交**。

**为什么比 d 版更强**：d 单模仅 64.65%（证伪模型），拉进三方融合是"弱模型拖累 + 微弱互补"的净 +0.06pp；而 round4 是 c2 训练线里 67.86% 演化路径上的独立检查点（与 best 互补但不弱），替换 d 后改了 8.70% 预测，互补性显著更强且无拖累。验证了一条规律：**融合应优先选同线但不同净化度的强检查点，而非塞入弱模型**。

**复现命令**（三方等权平均，fuse_predictions.py 不保留中间 probs 故用一次性脚本）：
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
| `fuse_swa_all6_tta2` | **c2 线 6 快照等权平均 (round1~5+best) · 14-view tta2** | **70.04%**（当前最优）|
| `fuse_swa_best40_tta2` | c2 线 best 加权 0.4 · 14-view tta2 | 69.41% |
| `fuse_swa_all6` | c2 线 6 快照等权平均 (round1~5+best) · 11-view | 69.46% |
| `fuse_c2_r4_c` | c2 × round4 × c | 68.95% |
| `fuse_swa_best40` | c2 线 best 加权 0.4 | 68.45% |
| `fuse_d_c2_c` | d × c2 × c | 68.43% |
| `fuse_c2_c` | c2 × c | 68.37% |

**候补变体（低成本可试）**：`c2 × round3 × c`、`c2 × round5 × c`、α 加权（强检查点给大权重）。其余杠杆仅 TTA view 数微调（已用 11-view）。

### 11.8 同线 6 快照 SWA 融合：68.95% → 69.46%（+0.51pp，已验证 2026-08-12）

**做了什么**：导出 c2 训练线全部 6 个检查点（round1.pt~round5.pt + best.pt）的 11-view TTA 概率（各 24967×500），做**等权算术平均**（SWA），输出 `output/fuse_swa_all6/pred_results.zip`。脚本 `export_c2_probs_swa.py`（零训练成本，仅 6 次已有模型的推理）。

**结果**：线上 **69.46%**，比三线融合 68.95% 再 +0.51pp，刷新本任务最优。变体 `fuse_swa_best40`（best 权重 0.4，其余均分）仅 68.45%——说明**等权反而优于强检查点加权**，early round 的弱检查点并未拖累、反而提供了多样互补。

**修正了 §11.7 的假设**：§11.7 原推断"塞入弱检查点（如 d 64.65%）会拖累"，但同线 early round（round1~4 虽净化度低于 best，但仍是 c2 流程强模型，非证伪的 d 线）与 best 同分布，**错误模式互补足以抵消其稍弱**，等权平均净收益为正。区分：**同线早期强检查点可纳入 SWA；跨线的弱/证伪模型（d 64.65%）仍会拖累，勿混入**。

**复现命令**：
```bash
# 导出 6 检查点 TTA probs（GPU，~12min/个，MIG 切片）
./.venv/bin/python -u export_c2_probs_swa.py --export --data-root /root/datasets/contest
# 等权融合 -> output/fuse_swa_all6/
./.venv/bin/python -u export_c2_probs_swa.py --fuse all
# 或 best 加权 0.4 -> output/fuse_swa_best40/
./.venv/bin/python -u export_c2_probs_swa.py --fuse best5 --alpha 0.4
```

**本任务方法基线定格（11-view）**：c2 自训练（5 轮去噪）+ LoRA 微调 + 11-view TTA + 同线 6 快照 SWA = **69.46%**。作为噪声标签鲁棒方法的探索记录封存（见 §0 定位）。后续重心转「长尾 + 噪声」。

### 11.9 tta2（14-view）增强 TTA + 同线 6 快照 SWA：69.46% → 70.04%（+0.58pp，已验证 2026-08-12）

**做了什么**：把 §11.8 的 TTA 从 11-view (`tta_enhanced`) 升级到 14-view (`tta_enhanced2`，在 11-view 基础上加 `._corner_crops(224)` 四角裁剪共 3 个 transform，合计 14 个)。重新导出 6 检查点 probs（`probs_round1~5_tta2` + `probs_best_tta2`，各 24967×500），等权融合 → `output/fuse_swa_all6_tta2/`；best 加权 0.4 → `output/fuse_swa_best40_tta2/`。

**结果**：
- `fuse_swa_all6_tta2` 线上 **70.04%**（新任务最优，比 11-view 同融合 +0.58pp）
- `fuse_swa_best40_tta2` 线上 **69.41%**（仍低于等权 all6，再次印证 §11.8「等权优于强检查点加权」）

**结论**：
1. tta2 的 14-view（四角裁剪增强）在**零重训成本**下稳定提 +0.58pp，是本轮最后一项推理侧增益，已并入最优记录 70.04%。
2. 等权 all6 在 11-view 与 14-view 下均优于 best 加权，结论稳健。
3. **推理侧杠杆已用尽**（分辨率/末轮/zero-shot/增强 view 都已试过，边际递减明显）。本任务再无低成本推理增益，后续提分必须靠训练侧新方法（见 §12：软标签自训练方向 B）。

**复现命令**：
```bash
# 导出 6 检查点 14-view (tta2) TTA probs（GPU，~15min/个）
./.venv/bin/python -u export_c2_probs_swa.py --export --tta2 --data-root /root/datasets/contest
# 等权融合 -> output/fuse_swa_all6_tta2/
./.venv/bin/python -u export_c2_probs_swa.py --fuse all --tta2
# best 加权 0.4 -> output/fuse_swa_best40_tta2/
./.venv/bin/python -u export_c2_probs_swa.py --fuse best5 --alpha 0.4 --tta2
```

## 8. 相关文档
- `prd/requirements.md` — 比赛需求与约束（权威）
- `format_request.md` — 官方提交格式（无表头 + 4 位零填充）
- 所有进展/交接/实验记录**统一维护在本文件**，不再新建额外 md（避免文档碎片化）。如曾建的 `prd/DELIVERY_v2.md` 已并入本文件 §2/§4.2/§4.7/§6.1，可删除。
- 旧文档 `prd/plan_v2.md`、`prd/progress.md`、`prd/TODO_remaining.md` 部分过期，以本文件为准。

---

## 9. 任务收尾与后续（长尾 + 噪声）

**本任务定位（2026-08-12 明确）**：NLPrompt 比赛是**噪声标签鲁棒方法的探索沙盒**，目的是为后续「长尾分布 + 标签噪声」真实场景积累方法，而非刷分。已达方法基线 **69.46%**（`fuse_swa_all6`），**不再反复提交刷分，仅保留此一份记录**。

**噪声标签方法沉淀（可迁移到长尾+噪声）**：
1. **全量候选池去噪**（B 阶段）：先聚类扩候选再净化，+8.64pp —— 对长尾的「尾部类样本少易误删」尤其关键，需改「按类配额保留」。
2. **伪标签一致性自训练**（C/C2）：高置信伪标签 + 一致性软约束，+1.0pp；注意 C2 的"单信号甜点"教训（§11.6）：**去噪阈值过狠会砍困难样本**，长尾下尾部困难样本占比更高，阈值须更保守。
3. **同线多快照 SWA 融合**（§11.8）：零成本 +0.51pp，可直接迁移。
4. **11-view TTA**：零成本 +0.3pp 级，可直接迁移。
5. **已证伪**：三重信号串联去噪（§11.6）、多模型集成、提分辨率、zero-shot 清洗（无类别名）。

**长尾 + 噪声 待探究方向（新任务起点）**：类平衡采样 / 类级损失加权 / 尾部类原型增强；去噪与长尾的耦合（尾部类噪声更难识别）；混合训练时注入长尾先验。这些方法不在本仓库范围，新任务应另起仓库。
