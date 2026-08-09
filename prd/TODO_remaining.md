# Contest 项目：剩余待办 & 可复制命令

> 生成时间：2026-08-01
> 当前状态：阶段 1 已完成，阶段 2 训练进行中（PID 1821325，约 20 分钟跑完 200 epoch）。
> 所有命令均从项目根目录 `/root/code/NLPrompt` 执行，GPU 任务必须用 `./.venv/bin/python`。
> 后台长任务统一用 `setsid nohup ... > xxx.log 2>&1 < /dev/null &` 脱离终端，防止 SSH 断开后进程被杀。

---

## 已完成（无需再做）

- ✅ 阶段 1：类内 K-means(k=3) 去噪，产物 `output/contest_prototype/`（features / prototypes / clean_mask 等齐全，保留 54.8% 干净样本）
- ✅ 损坏图清洗：`output/contest_clean/` 已存在（clean_train_manifest.json / clean_test_manifest.json 齐全）
- ✅ 阶段 2 代码实现完成，且**训练已启动**（PID 1821325）

---

## 你需要做的事（按顺序）

### 步骤 0：确认阶段 2 训练已跑完（必须先做）

阶段 2 训练约 20 分钟。检查是否结束、产物是否生成：

```bash
# 查看训练进度
tail -n 5 /root/code/NLPrompt/output/train_prototype.log

# 确认产物 best.pt 已生成（训练结束的标志）
ls -la /root/code/NLPrompt/output/contest_train/best.pt
```

> 若 `best.pt` 还没出现，说明训练未结束，等它跑完再继续下面的步骤。

---

### 步骤 1：阶段 3 自训练（扩展干净样本 + 重训）

直接复用脚本 `scripts/run_selftrain.sh`，它会一次性完成：
- 3a：用阶段 2 的 best.pt 对训练图打伪标签、捡回高置信度样本，扩展 clean mask
- 3b：在扩展后的干净集上重训 100 epoch
- 3c：用 TTA 推理并生成 `pred_results.zip`

```bash
cd /root/code/NLPrompt
setsid nohup bash scripts/run_selftrain.sh > output/selftrain.log 2>&1 < /dev/null &
echo "自训练流水线已启动，查看输出: tail -f output/selftrain.log"
```

跑完的标志：
```bash
ls -la /root/code/NLPrompt/output/contest_train_final/pred_results.zip
```

> 预计耗时：扩展 mask（几分钟）+ 重训 100 epoch（约 10 分钟）+ 推理（几分钟）。

---

### 步骤 2：若想单独跑/重跑推理（阶段 4，TTA）

如果只想用阶段 2 的 `best.pt` 直接出提交包（跳过自训练），或自训练后又想换模型出包：

```bash
cd /root/code/NLPrompt
setsid nohup ./.venv/bin/python -u test_prototype.py \
    --checkpoint output/contest_train/best.pt \
    --data-root /root/datasets/contest \
    --clean-test-manifest output/contest_clean/clean_test_manifest.json \
    --clip-weights /root/weights/ViT-B-32.pt \
    --output-dir output/contest_train \
    --tta \
    > output/test_prototype.log 2>&1 < /dev/null &
```

自训练后的最终包则用：
```bash
setsid nohup ./.venv/bin/python -u test_prototype.py \
    --checkpoint output/contest_train_final/best.pt \
    --data-root /root/datasets/contest \
    --clean-test-manifest output/contest_clean/clean_test_manifest.json \
    --clip-weights /root/weights/ViT-B-32.pt \
    --output-dir output/contest_train_final \
    --tta \
    > output/test_prototype_final.log 2>&1 < /dev/null &
```

产物为 `output/contest_train/pred_results.zip` 或 `output/contest_train_final/pred_results.zip`。

---

## 关键产物路径速查

| 内容 | 路径 |
|------|------|
| 阶段 1 原型/去噪 | `output/contest_prototype/`（features.pt, prototypes.pt, clean_mask.pt, kept_idx.json, prototype_info.json, sample_confidence.pt） |
| 阶段 1 编码日志 | `output/class_prototype.log` |
| 干净样本 manifest | `output/contest_clean/clean_train_manifest.json`、`clean_test_manifest.json` |
| 阶段 2 训练产物 | `output/contest_train/best.pt`、`ema_final.pt` |
| 阶段 2 训练日志 | `output/train_prototype.log` |
| 阶段 3 扩展 mask | `output/contest_prototype_selftrain/` |
| 阶段 3 重训产物 | `output/contest_train_final/best.pt` |
| 最终提交包 | `output/contest_train_final/pred_results.zip` |

---

## 注意事项

1. **不要重启阶段 1**：`output/contest_prototype/` 已完整，重跑 `class_prototype.py` 或 `run_pipeline_prototype.sh` 会浪费约 3 小时重复编码。
2. **CLIP 权重**：固定用 `/root/weights/ViT-B-32.pt`，禁止外部数据/换模型（比赛约束）。
3. **提交格式**：比赛要求 `pred_results.zip`，由 `test_prototype.py` 生成，包含模型输出索引 → 4-digit folder ID 的映射。
4. **后台任务防断连**：所有长任务务必带 `setsid nohup ... < /dev/null &`，否则 SSH 一断进程就没了。
