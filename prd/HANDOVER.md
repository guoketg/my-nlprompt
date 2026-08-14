# NLPrompt Contest 项目 — AI 接手交接文档（主控/索引）

> 更新时间：2026-08-14（方向 B 软标签已验收证伪；tta2 14-view SWA 融合 70.04% 为当前最优）
> 项目根目录：**/root/code/NLPrompt**
> 本任务定位：比赛目的为**探索噪声标签鲁棒方法**，为后续「长尾 + 噪声」场景打基础，**不追求反复刷分**，保留方法记录即可。

## 子文档导航（点击跳转）

| 主题 | 文件 | 内容 |
|------|------|------|
| 现状 / 历史分数线 / 进行中 | [handover/01-status.md](handover/01-status.md) | §0 现状速记、分数线、关键认知 |
| 方案演进 / 阶段 A 归档 | [handover/02-evolution.md](handover/02-evolution.md) | §2 演进表、阶段 A 废弃记录 |
| 已踩的坑 | [handover/03-pitfalls.md](handover/03-pitfalls.md) | §4.1~4.7 提交格式/虚假val_acc/分辨率/LoRA/NaN/device/权重加载 |
| 流水线命令 / 各阶段结果 | [handover/04-pipeline.md](handover/04-pipeline.md) | §5 训练推理自训练融合命令、§6 各阶段归档 |
| 关键认知 / 排除方向 / 时间铁律 | [handover/05-cognition.md](handover/05-cognition.md) | §9 认知、§9.5 排除方向、§10 时间预估 |
| 经验总结（突破记录） | [handover/06-learnings.md](handover/06-learnings.md) | §11.1~11.9 全部阶段经验与融合对比 |
| 后续方向（软标签 + 长尾噪声） | [handover/07-next.md](handover/07-next.md) | §12 软标签进行中、§13 长尾+噪声收尾 |

---

## 0. 一句话现状（详情见 [01-status](handover/01-status.md)）

- **线上准确率当前最优 = 70.04%**（`output/fuse_swa_all6_tta2/pred_results.zip` = c2 线 6 检查点 round1~5+best 的 **14-view (tta2) TTA 概率**等权平均，2026-08-12 官网确认）。11-view 同融合 69.46% 作方法基线已封存。
- **方向 B 软标签自训练已验收、证伪**：出包 67.08% / 66.49%（11-view TTA 公平对比 c2 单模 67.86% / 融合 68.95%），**低于 c2**，方向不成立。详见 [06-learnings §11.10](handover/06-learnings.md#1110-方向-b软标签证伪2026-08-14)、[07-next §12](handover/07-next.md#12-方向-b软标签已验收证伪)。
- **推理侧杠杆已用尽**（分辨率/末轮/zero-shot/增强 view 都已试过，边际递减明显），后续提分靠训练侧新方法（方向 C 难样本加权不删，见 [07-next §13.1](handover/07-next.md#131-方向-c难样本加权不删待启动)）。

---

## 1. 任务与约束（权威以 `prd/requirements.md` 为准）

- **任务**：500 类细粒度图像分类，训练图网络爬取、类内噪声严重，无可靠类别名。
- **硬约束**：① 骨干必须 CLIP **ViT-B/32**（`/root/weights/ViB-32.pt`，不得替换）② 不得引入外部数据集 ③ 禁止多模型集成/投票 ④ 可复现 ⑤ 人工清洗非必要前置。
- **数据**：训练 103,218 张（`/root/datasets/contest/train/0000...0499/`）；测试 24,967 张（`/root/datasets/contest/test/*.jpg`，无标签）。
- **提交格式**：`pred_results.csv` 无表头，每行 `图片文件名,类别编号`（4 位零填充），打包 `pred_results.zip`。详见 [03-pitfalls §4.1](handover/03-pitfalls.md#41-提交格式无表头--4-位零填充)。

---

## 3. 产物速查（路径 / 线上分）

| 内容 | 路径 | 线上 |
|------|------|------|
| **当前最优提交（14-view SWA）** | `output/fuse_swa_all6_tta2/` | **70.04%** |
| 11-view 同融合（方法基线） | `output/fuse_swa_all6/` | 69.46% |
| 扩展融合（c2×r4×c） | `output/fuse_c2_r4_c/` | 68.95% |
| c2 阶段单模 | `output/contest_ft_lora_c2/` | 67.86% |
| 方向 B 软标签（证伪）| `output/contest_ft_lora_c2_soft/` | 67.08% / 66.49% |
| 阶段 C / B / 旧V2 / 阶段A(废弃) | `output/contest_ft_lora_c` / `_b` / `` / `_a` | 64.66% / 63.66% / 55.02% / 34%(勿提交) |
| V2 训练/推理脚本 | `train_clip_lora.py` / `test_clip_lora.py` | — |
| 自训练脚本 | `self_train_v2.py`（含 `--soft-labels`）、`self_train_v3.py`（证伪）| — |
| 融合/导出脚本 | `fuse_predictions.py` / `export_c2_probs_swa.py` | — |

> `all_class_predictions.json` 是 API 识别结果、类别名重叠错误多，**不可作监督信号**；但 `train_clip_lora.py` 仍读取它（仅类别数对齐），**删除会崩溃**。

---

## 7. 环境铁律

- **GPU 任务必须用 `./.venv/bin/python`**（torch 2.6.0+cu124，H200 MIG 2g.35gb）。系统 `python3` 是别的项目，勿用。
- 后台长任务固定模板：`setsid nohup ./<脚本> ... > xxx.log 2>&1 < /dev/null &`（否则 SSH 断开进程被杀）。
- CLIP 权重 `/root/weights/ViT-B-32.pt` 加载约 3 分钟无输出，属正常。
- 启动长训练前**务必先验证 GPU 可见**：`./.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`，否则会静默回退 CPU（软标签进程曾踩此坑，见 [03-pitfalls 补充坑](handover/03-pitfalls.md)）。
- 导出前确认根目录有 `contest_clip.py`（曾被删，原文件在 `legacy/contest_clip.py`），否则 `ModuleNotFoundError`。

---

## 8. 相关文档

- `prd/requirements.md` — 比赛需求与约束（权威）
- `format_request.md` — 官方提交格式
- 所有进展/交接/实验记录统一维护在本主控 + `handover/` 子文档，不再新建额外 md。
