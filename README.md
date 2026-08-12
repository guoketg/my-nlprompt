# NLPrompt — 含噪标签细粒度识别竞赛

基于 **CLIP ViT-B/32** 在标签含噪网络图像（500 类自然动植物）上做鲁棒微调，目标线上准确率 75%（当前最优 68.95%）。

## 硬约束（赛题规定）

- 骨干固定为 CLIP `ViT-B/32`（224px，不可提升输入分辨率）。
- 单模型，禁止多模型集成；无类别名、无外部数据；权重固定为 `/root/weights/ViT-B-32.pt`。
- 提交格式：无表头 `pred_results.csv`，每行 `filename,label`（label 为 4 位补零类号），压缩为 `pred_results.zip`。
- **唯一可信指标是官网提交分**；本地 val_acc 因标签含噪而虚假（不要据此判断）。

## 环境

```bash
# 本项目 GPU 任务必须用自有 venv（torch 2.6.0+cu124，H200 MIG 2g.35gb）
./.venv/bin/python script.py ...
```
长任务后台启动（脱离 SSH 会话）：
```bash
setsid nohup ./.venv/bin/python -u script.py ... > run.log 2>&1 < /dev/null &
```

## 当前最优方法（C2 自训练线）

纯视觉、去噪 + LoRA 微调 + 多检查点融合：

1. **去噪自训练**（`self_train_v2.py`）：5 轮激进自训练，每轮在 CLIP 提特征上 K-means 去噪，仅保主导物种，`uncertain` 权重 0.05，`mismatch` 直接丢弃。`output/contest_ft_lora_c2/` 产出 `round1~5.pt` + `best.pt`。
2. **LoRA 微调**（`train_clip_lora.py`）：在去噪样本上 LoRA 微调 CLIP 视觉骨干（目标层 `out_proj/c_fc/c_proj`，rank=8，warmup 5 epoch）。
3. **增强 TTA 推理**（`test_clip_lora.py`）：11-view（多尺度 1.0/1.33 + 角落 + 翻转 + 轻微 ColorJitter）。
4. **多检查点 / 多线融合**（`fuse_predictions.py`、`export_c2_probs_swa.py`）：SWA 平均不同 round 的 probs，并与 C 线（stage-2 余弦分类器）动态置信加权融合。

> 当前最优：`c2×round4×c` 三方等权融合 = **68.95%**。

## 提交的脚本（当前活跃）

| 文件 | 作用 |
|------|------|
| `self_train_v2.py` | C2 去噪自训练（5 轮），产出 `output/contest_ft_lora_c2/` |
| `train_clip_lora.py` | LoRA 微调 CLIP 视觉骨干 |
| `test_clip_lora.py` | 增强 TTA 推理，导出 `pred_results.csv/.zip` |
| `fuse_predictions.py` | 多线（c2 / c）动态置信加权融合 |
| `export_c2_probs_swa.py` | 导出 C2 全部 checkpoint probs 并做 SWA 融合（冲刺 75% 用） |
| `request.md` / `format_request.md` | 赛题说明与提交格式 |

## 冲刺 75% 规划

见 [`prd/PLAN_TO_75.md`](prd/PLAN_TO_75.md)（方法修改 + 全新方案 X1 聚类重标等）。
当前进度与经验沉淀见 [`prd/HANDOVER.md`](prd/HANDOVER.md)。

## 目录

```
./
├── self_train_v2.py / train_clip_lora.py / test_clip_lora.py   # 当前主流程
├── fuse_predictions.py / export_c2_probs_swa.py                # 融合提交
├── prd/                # HANDOVER / PLAN / progress 文档
├── configs/            # Dassl 配置（datasets/trainers）
├── output/             # 训练产物与提交包（contest_ft_lora_c2 等）
├── clip/ Dassl.pytorch/ datasets/ trainers/                    # 框架依赖
└── legacy/             # 早期方案残留（纯视觉原型 / prompt 方案），已弃用
```

> `all_class_predictions.json`：早期 API 类别名识别结果，经验证不可靠且重叠，**已弃用**，请勿再使用。
