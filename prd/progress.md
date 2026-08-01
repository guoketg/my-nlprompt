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

### 阶段 1：类内聚类去噪（当前重点）
- [x] 实现 `class_prototype.py`：
  - 提取所有训练图 CLIP 视觉特征
  - 每类 K-means(k=3) 聚类
  - 识别主导簇并计算视觉原型
  - 输出干净样本 mask
- [x] 在 3 个类的小数据上验证通过：clean ratio 39.6% ~ 69.6%，符合预期
- [ ] 在完整 500 类数据上运行原型提取（后台任务 PID 1733828 进行中，batch 180/404，预计总耗时约 3 小时）
- [ ] 验证聚类质量：检查 500 类保留比例分布

### 阶段 2-4 代码已完成
- [x] 实现 `train_prototype.py`：可学习类别嵌入 + GCE + CE + EMA
- [x] 实现 `test_prototype.py`：推理 + TTA + 生成 pred_results.zip
- [x] 实现 `self_train_prototype.py`：伪标签扩展干净样本
- [x] 实现 `inspect_prototypes.py`：检查每类聚类纯度
- [x] 实现 `scripts/run_pipeline_prototype.sh`：阶段 1-2-4 端到端流水线
- [x] 实现 `scripts/run_selftrain.sh`：阶段 3 自训练流水线
- [x] 修改 `clean_contest.py`：`--json` 变为可选，不依赖 `all_class_predictions.json`
- [x] 3 类小数据端到端验证通过（prototype + train + inspect）
- [ ] 完成阶段 1（500 类）后联调训练与测试

## 待完成

### 阶段 2：分类器训练
- [ ] 实现 `train_prototype.py`：
  - 用阶段 1 原型初始化 500 个可学习类别嵌入
  - 不使用 JSON 类别名
  - GCE loss + CE 高置信度样本
  - Outlier exclusion + EMA 教师
- [ ] 实现 `test_prototype.py`：
  - 加载训练好的分类器
  - 映射模型输出索引 → 4-digit folder ID
  - 生成 `pred_results.csv` / `.zip`

### 阶段 3：迭代自训练
- [ ] 用当前模型对全部训练图打伪标签
- [ ] 捡回高置信度原标签样本
- [ ] 重新 warmstart 训练

### 阶段 4：推理增强
- [ ] TTA：多裁剪 + 水平翻转
- [ ] Soft-voting（单个模型多个 checkpoint 概率平均）

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
