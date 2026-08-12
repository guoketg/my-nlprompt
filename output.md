我先并行做三件事：查看两个概率导出任务是否完成、读取 HANDOVER.md 当前内容以便更新。
两个概率导出任务都已完成（c2 和 c 的 `pred_probs.npy` 均已生成，shape=(24967, 500)）。现在我：
1. 运行融合脚本产出候选提交 2
2. 更新 HANDOVER.md 记录 67.86% 成果
3. （3）推理已完成确认

先跑融合：
融合完成（候选 2 已产出 `output/fuse_c2_c/pred_results.zip`）。注意 A weight mean=0.500 说明 c2 和 c 在大部分样本上置信度接近，融合更多是互补而非主导——c2 是更强的模型，我用 --alpha 0.6 让 c2 略占优但融合逻辑按相对置信动态加权。

现在更新 HANDOVER.md。我需要把 §6 待办、§11 进度、以及 67.86% 的新基线都更新进去。先读 §11 部分：
现在我已经掌握全文结构。我将更新以下部分：
1. §6 待办 → 记录 c2 已达 67.86% 为新基线，并新增 §6.3 阶段 C2
2. §11 新增 §11.4 阶段 C2 记录（5轮激进自训练+增强TTA=67.86%）
3. §10.4 模板补充 c2 与融合预估

先更新 §6 头部和待办：
Now add §6.3 (阶段 C2) after §6.2:
Now add §11.4 (阶段 C2 经验总结) after §11.3, and update §11.2 heading reference. Let me add §11.4:
Now update §10.4 template to add c2 and fusion estimates, and §9.2 ceiling mention. Let me update §10.4:
Now update §9.2 ceiling assessment to reflect the new 67.86% high-water mark:
Let me verify the HANDOVER edits are consistent and check no broken anchors. Quick read of updated §6 head + §9.2 + §10.4:
HANDOVER.md 已更新完成，无 lint 错误。所有三个任务都已处理：

## ✅ (1) HANDOVER 已更新
新增/修订：
- **§6** 待办头部：新基线 = c2 **67.86%**；新增 §6.3 阶段 C2 记录
- **§9.2** 天花板评估：补入 67.86%
- **§10.4** 预估模板：补 c2 训练、增强 TTA、概率导出、融合的时长
- **§11.4**（全新）：阶段 C2 经验总结——激进剔除 mismatch 比降权更彻底、放松阈值 + uncertain 扩大监督、增强 TTA 零成本必带、本地 val 仍不映射线上等教训

## ✅ (2) 方向 4 已确认纳入规划
§6 第 6 条已写明"方向 4 — 三重信号串联自训练"作为下一候选，结构性优化、~10h 训练、收益不确定。

## ✅ (3) 推理已完成确认
两个概率导出任务（c2、c）均已完成，`pred_probs.npy` (24967×500) 已生成；融合脚本已跑出 **候选提交 2**：`output/fuse_c2_c/pred_results.zip`（c2 × c per-class 置信度加权融合，c2 权重略占优）。

---

## 当前可提交候选包

| 候选 | 包路径 | 状态 | 预期 |
|---|---|---|---|
| **候选 1** | `output/contest_ft_lora_c2/pred_results.zip` | ✅ **已交，线上 67.86%** | 当前最优 |
| **候选 2** | `output/fuse_c2_c/pred_results.zip` | 🔄 已产包，待你提交验收 | 可能 +0~0.5pp |

**下一步建议**（你已认同方向 4 值得做）：
- 先提交候选 2（融合包）看是否超 67.86%
- 同时我可以**启动方向 4 训练**（三重信号串联，~10h，过夜跑），不阻塞你提交

要我现在就启动方向 4 训练吗？还是先等你提交候选 2 的线上结果再决定？