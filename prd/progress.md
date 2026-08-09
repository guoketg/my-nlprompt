# Contest 项目进度

## 已完成

### 1. 环境与数据盘点
- [x] 确认训练数据规模：500 类，103,218 张训练图，24,967 张测试图
- [x] 确认比赛约束：必须用 CLIP ViT-B/32、禁止模型集成、禁止外部数据
- [x] 确认 `all_class_predictions.json` 不可靠，决定不依赖文本类别名

### 2. 代码梳理
- [x] 阅读 `clean_contest.py`：已支持损坏/退化图片检测与隔离
- [x] 阅读 `train_contest.py`：现有 PromptLearner + GCE loss + outlier exclusion，依赖 JSON 类名
- [x] 阅读 `test_contest.py`：现有推理依赖 JSON 类名
- [x] 阅读 `datasets/contest.py`：数据加载逻辑已了解

### 3. 方案规划
- [x] 确定方向：纯视觉聚类去噪 + 可学习类别嵌入分类器
- [x] 确定阶段：
  1. 类内 K-means 聚类去噪
  2. 可学习类别嵌入训练（cosine classifier）
  3. 迭代自训练（伪标签）
  4. 测试时增强（TTA）

### 4. 项目文档整理
- [x] 创建 `prd/requirements.md`
- [x] 创建 `prd/progress.md`
- [x] 清理不再需要的临时/旧文件

## 进行中

无。全部阶段已完成。

### 阶段 1：类内聚类去噪 ✅ 完成
- [x] 实现 `class_prototype.py`：提取 CLIP 特征 + 每类 K-means(k=3) 聚类 + 去噪 + 原型
- [x] 3 类小数据验证通过：clean ratio 39.6% ~ 69.6%
- [x] 完整 500 类原型提取完成（2026-08-01 01:27 落盘，保留 54.8% 干净样本 = 56,565/103,218）
- [x] 聚类质量已通过后续训练/推理间接验证

### 阶段 2-4 全部完成 ✅
- [x] 实现 `train_prototype.py`：可学习类别嵌入 + GCE + CE + EMA（阶段 2 产物 `output/contest_train/best.pt`，200 epoch）
- [x] 实现 `test_prototype.py`：推理 + TTA + 生成 pred_results.zip
- [x] 实现 `self_train_prototype.py`：伪标签扩展干净样本
- [x] 实现 `inspect_prototypes.py`：检查每类聚类纯度
- [x] 实现 `scripts/run_pipeline_prototype.sh`、`scripts/run_selftrain.sh`
- [x] 修改 `clean_contest.py`：`--json` 变为可选
- [x] 阶段 3 自训练重训完成：`output/contest_train_final/best.pt`（100 epoch，覆盖 496/500 类）
- [x] 阶段 4 推理完成：两份 `pred_results.csv/.zip` 已生成（2026-08-07 03:49）
- [x] 修复 csv 格式坑（去 `test/` 前缀 + 去掉表头 + label 改 4 位零填充，符合 `format_request.md`），提交包已重新打包

## 待完成

### 阶段 2：分类器训练 ✅ 代码已完成并运行
- [x] `train_prototype.py`：原型初始化 500 嵌入 + GCE + CE + Outlier exclusion + EMA
- [x] 产物 `output/contest_train/best.pt`（200 epoch）

### 阶段 3：迭代自训练 ✅ 已完成
- [x] 伪标签扩展干净样本 + 重训 100 epoch
- [x] 产物 `output/contest_train_final/best.pt`（覆盖 496/500 类）

### 阶段 4：推理增强 ✅ 已完成
- [x] TTA 推理 + 生成 `pred_results.csv/.zip`
- [x] 修复 csv 格式（去 `test/` 前缀、去表头、label 改 4 位零填充字符串，符合官方 `format_request.md`），提交包已重新打包

### 当前可交付物
- [x] `output/contest_train_final/pred_results.zip`（推荐提交，阶段 3 模型）
- [x] `output/contest_train/pred_results.zip`（阶段 2 模型备选）
- [ ] 可选：Soft-voting 多 checkpoint 概率平均再出包（精度可能更高）

## 废弃/不再使用

| 文件/模块 | 废弃原因 |
|-----------|----------|
| `all_class_predictions.json` | API 识别结果不可靠，不再作为类别名输入 |
| `class_to_species_optimized.json` | 旧文本侧映射，与纯视觉方案无关 |
| `clip_words.csv` | 文本提示词库，当前方案不使用 |
| 旧 `contest_clip.py` 的 PromptLearner | 依赖文本类名，不再使用 |
| `command.md` | Stanford Cars 旧命令记录，与本项目无关 |
| `DELIVERY.md` | 旧交付文档，已过期 |
| `stanford_cars_train.log` | 旧训练日志，与本项目无关 |

## 关键决策记录

1. **不使用 `all_class_predictions.json`**
   - 决策人：用户
   - 原因：该文件为 API 识别结果，类别名重叠/错误多，不可信
   - 影响：从 Prompt Learning 转向纯视觉原型学习

2. **保留 `clean_contest.py` 的损坏图过滤**
   - 原因：截断/退化图检测与当前方案不冲突，且是必要的数据清洗

3. **使用 K-means(k=3) 做类内去噪**
   - 原因：每个文件夹约 207 张图，大概率包含 2-3 种不同概念；最大簇作为主导物种
