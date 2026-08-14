# NLPrompt Contest — 已踩的坑（PITFALLS）

> 返回 [主控 HANDOVER](../HANDOVER.md)

## 4.1 提交格式：无表头 + 4 位零填充

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

## 4.2 本地 val_acc 是虚假指标（关键！）

- 训练里的 `val_acc` 是从阶段 1 `clean_mask`（被判定"干净"的样本）切出的，且 train/val 同源于此净化池，标签是文件夹名（非人工核实）。模型在"精选干净子集"上自然虚高（曾达 0.82）。
- 动态筛选一收紧（候选从 56,565 砍到 ~12,313，约 22%），val_acc 立刻跌到 ~0.66。这进一步证明它测的是"对最容易样本的拟合"，**不代表线上混合噪声测试集的泛化**。
- **唯一可信指标 = 官网提交分**。本地 80% 切勿外推。判断是否提分只能靠提交。
- **warmup 模型反而泛化更好（2026-08-09 实测）**：warmup 期（epoch3-5，全量训练、未过拟合筛选子集）val_acc 虚高达 0.78，但提交后线上 55%；末轮（epoch40，loss=0.016、过拟合筛选子集）提交后线上仅 34%。**选提交模型应 early-stop / 取 warmup 期，而非末轮或 max-val_acc**。本项目的 `best.pt` 选择逻辑（`--keep-final` 保存末轮）已被证伪，应改用 `--keep-best-val` 或显式取 warmup 检查点。

## 4.7 CLIP ViT-B/32 仅支持 224 分辨率（实测误述修正）

- 旧文档 §6 曾写"提高输入分辨率到 336/448" 可提分 —— **错误，不可行**。ViT-B/32 的 `positional_embedding` 是按 224 输入（50 个位置：7×7 patch + 1 cls）训练的。
- 实测：用 `--resolution 336` 前向报 `positional_embedding: expected 101 but got 50`（336 → 16×16=256 token，与预训练 50 位置不符），直接报错。
- **结论**：分辨率固定 224，提分不能靠分辨率，只能靠筛选策略 / 候选池 / 早停 / 自训练迭代。

## 4.3 CLIP LoRA 注入：必须子类化 nn.Linear

- CLIP 的 `ResidualAttentionBlock` 用 `nn.MultiheadAttention`，其 `out_proj` 在注意力内部直接访问 `.weight`。若把 `out_proj` 替换成外部包裹的 LoRA 模块（无 `.weight` 属性），前向会 `AttributeError: ...has no attribute 'weight'`。
- **修复**：`_LoRALinear` 继承 `nn.Linear`（`super().__init__` 保留 `.weight/.bias`），forward 调 `super().forward(x) + lora_branch`。这样既能注入 LoRA，又不破坏 MHA 内部访问。
- CLIP 的 `in_proj_weight` 是合并 QKV 权重（非独立 Linear 模块），本项目实现下暂**未**微调（仅微调 `out_proj/c_fc/c_proj`）。

## 4.4 动态筛选 compute_losses 的数据泄漏/NaN

- 全量候选里有损坏图，DataLoader 加载失败被 `_collate` 过滤 → 这些样本 loss 保持 NaN。
- 原 `select_small_loss_classwise` 遇 NaN 直接 `raise`（曾崩在 `Missing loss for class 274`）。**修复**：`compute_losses` 记录实际加载 idx，把未加载样本从候选池永久剔除；筛选函数遇 NaN 改为跳过该类（不抛异常）；`combine_loss_proto` 用 `nanargmin` 安全取最小 loss 下标。

## 4.5 推理脚本 device 不匹配 / TTA flip

- `test_clip_lora.py` 中 `inject_lora` 在 CPU 创建 LoRA 参数，须在 `inject_lora` 后补一次 `model.to(device)`，否则 base weight（cuda）与 LoRA（cpu）device 不一致报错。
- TTA 用 `T.RandomHorizontalFlip()`，不要用 `T.Lambda(lambda x: x.flip(-1))`（PIL Image 无 `flip` 方法）。

## 4.6 CLIP 权重加载慢

- `/root/weights/ViT-B-32.pt` 加载约 3 分钟无输出，属正常，勿以为卡死。

## 补充坑（2026-08-13 新增）

- **`contest_clip.py` 缺失导致导出崩溃**：`export_c2_probs_swa.py` 经 `import test_clip_lora` → `import contest_clip`，而根目录 `contest_clip.py` 曾被删（原文件在 `legacy/contest_clip.py`）。导出前务必确认根目录有 `contest_clip.py`，否则 `ModuleNotFoundError`。
- **软标签训练未在 GPU 上跑**：启动时若 `torch.cuda.is_available()` 返回 False（MIG 设备短暂不可用），`self_train_v2.py` 会静默回退 CPU，单轮耗时从 ~4h 暴涨。启动长训练前务必先 `python -c "import torch; print(torch.cuda.is_available())"` 验证 GPU 可见。
