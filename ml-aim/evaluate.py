"""
evaluate.py — 完整评估套件

包含：
  1. run_linear_probe()          — 线性探测（冻结编码器，量化表征质量）
  3. run_finetune_few_shot()     — 少标签微调（1% / 10% 标注量，核心实验）
  4. run_ablation()              — 消融实验（移除文本模态 / 减少标注量）
  5. run_baseline_imagenet()     — 对比基线（ImageNet 迁移）

每个函数均返回 dict 结果，便于 main.py 汇总打印。

评估指标说明：
  - Accuracy:    整体预测正确率
  - Sensitivity: micro-average sensitivity（one-vs-rest，等价于 micro-recall）
  - Specificity: micro-average specificity（one-vs-rest，见 micro_metrics_one_vs_rest）

修复记录：
  [BUG-4] run_linear_probe 中 `if not train_ds` 对 Dataset 对象做布尔判断，
          PyTorch Dataset 未实现 __bool__，会引发 NotImplementedError 或
          始终为 True，导致空数据集检查形同虚设。
          → 改为 `if len(train_ds) == 0 or len(val_ds) == 0`。
  [BUG-6] _print_report 中 docstring 排在 `if` 语句之后，
          Python 不会将其识别为函数文档字符串。
          → 将 docstring 移至函数体第一行。
"""
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score

from config import (
    DEVICE, BATCH_SIZE, LINEAR_PROBE_EPOCHS, FINETUNE_EPOCHS,
    LINEAR_PROBE_LR, FINETUNE_LR, WEIGHT_DECAY,
    NUM_CLASSES, OCT_IMAGE_ROOT, PHOTO_SAVE_DIR
)
from datasets  import LabeledOCTDataset, TRUE_CLASSES
from models    import (LinearProbeClassifier, FineTuneClassifier,
                       AIMv2OCTAnomalyDetector)
from utils     import (plot_accuracy_curves, plot_confusion_matrix,
                        plot_roc_curves, plot_pr_curves)


# =====================================================================
# 评估指标：micro-average sensitivity & specificity（one-vs-rest）
# =====================================================================
def micro_metrics_one_vs_rest(cm: np.ndarray):
    """
    计算 one-vs-rest 策略下的
      - micro-average sensitivity (micro-recall)
      - micro-average specificity

    cm: shape (C, C) 的混淆矩阵，cm[i,j] = 真实 i 预测 j

    返回 (micro_sens, micro_spec)
    """
    C = cm.shape[0]
    tp_sum = 0   # Σ TP_i
    fn_sum = 0   # Σ FN_i
    fp_sum = 0   # Σ FP_i
    tn_sum = 0   # Σ TN_i

    for i in range(C):
        tp_i = cm[i, i]                          # 真实 i 预测 i
        fn_i = cm[i, :].sum() - tp_i             # 真实 i 但预测非 i
        fp_i = cm[:, i].sum() - tp_i             # 预测 i 但真实非 i
        tn_i = cm.sum() - (tp_i + fn_i + fp_i)  # 其余三种之外的样本

        tp_sum += tp_i
        fn_sum += fn_i
        fp_sum += fp_i
        tn_sum += tn_i

    micro_sens = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    micro_spec = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) else 1.0
    return float(micro_sens), float(micro_spec)


# =====================================================================
# 通用分类训练 / 验证循环
# =====================================================================
def _train_classifier(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader:   DataLoader,
                      epochs:       int,
                      lr:           float,
                      tag:          str = "Classifier",
                      patience:     int = 5) -> Dict[str, Any]:
    """
    通用分类头训练循环。
    只优化 requires_grad=True 的参数（编码器冻结时只更新分类头）。

    修复：
      - 使用类别权重 CrossEntropyLoss，缓解 CNV 类主导导致的坍缩问题。
      - 使用 AdamW + 分层学习率衰减（编码器解冻层 lr×0.1）。
      - 加入 early stopping（连续 5 epoch val_acc 不提升则停止）。
      - 新增：记录每 epoch 的 sensitivity / specificity（基于混淆矩阵）。
    """
    # ── 计算类别权重（反频率加权）──────────────────────────────────────
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy().tolist())
    label_tensor = torch.tensor(all_labels)
    num_classes  = int(label_tensor.max().item()) + 1
    counts       = torch.bincount(label_tensor, minlength=num_classes).float()
    weights      = (counts.sum() / (num_classes * counts)).to(DEVICE)
    print(f"  [{tag}] 类别权重: { {i: f'{w:.3f}' for i, w in enumerate(weights.tolist())} }")
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── 分层学习率：编码器解冻层 lr×0.1，分类头 lr×1.0 ───────────────
    head_params    = []
    encoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("classifier", "head", "start_token",
                                    "decoder", "recon_head")):
            head_params.append(p)
        else:
            encoder_params.append(p)

    param_groups = [{"params": head_params,    "lr": lr},
                    {"params": encoder_params, "lr": lr * 0.1}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-3)

    best_val_acc   = 0.0
    best_preds:    list = []
    best_labels_:  list = []
    best_probs:    Optional[np.ndarray] = None
    best_sens:     float = 0.0
    best_spec:     float = 0.0
    train_accs, val_accs = [], []
    val_senss, val_specs = [], []
    no_improve = 0
    EARLY_STOP_PATIENCE = patience   # 由调用方传入，默认5

    for ep in range(1, epochs + 1):
        # ── 训练 ────────────────────────────────────────────────────
        model.train()
        correct, total = 0, 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"[{tag}] Epoch {ep}/{epochs} Train",
                                  leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += len(labels)
        train_acc = correct / total
        train_accs.append(train_acc)
        scheduler.step()

        # ── 验证 ────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels_ep, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(DEVICE)
                logits = model(imgs)
                probs  = torch.softmax(logits, 1).cpu().numpy()
                preds  = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels_ep.extend(labels.numpy())
                all_probs.append(probs)

        probs_np = np.concatenate(all_probs, axis=0)
        val_acc  = np.mean(np.array(all_preds) == np.array(all_labels_ep))
        val_accs.append(val_acc)

        # ── 计算 sensitivity / specificity ──────────────────────────
        cm = sk_confusion_matrix(all_labels_ep, all_preds,
                                  labels=list(range(num_classes)))
        sens, spec = micro_metrics_one_vs_rest(cm)
        val_senss.append(sens)
        val_specs.append(spec)

        print(f"  [{tag}] Epoch {ep}: "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"sens={sens:.4f}  spec={spec:.4f}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_preds    = all_preds
            best_labels_  = all_labels_ep
            best_probs    = probs_np
            best_sens     = sens
            best_spec     = spec
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  [{tag}] Early stopping at epoch {ep} "
                      f"（{EARLY_STOP_PATIENCE} epochs 无提升）")
                break

    return {
        "best_val_acc":  best_val_acc,
        "best_sens":     best_sens,
        "best_spec":     best_spec,
        "best_preds":    best_preds,
        "best_labels":   best_labels_,
        "best_probs":    best_probs,
        "train_accs":    train_accs,
        "val_accs":      val_accs,
        "val_senss":     val_senss,
        "val_specs":     val_specs,
    }


def _print_report(result: Dict, tag: str, class_names=None):
    """
    打印分类报告 + AUC + sensitivity/specificity。

    [FIX-6] 原版 docstring 写在 `if class_names is None` 之后，
            Python 不将其识别为函数文档字符串（只是孤立字符串表达式）。
            已移至函数体第一行，成为合法 docstring。
    """
    if class_names is None:
        class_names = TRUE_CLASSES

    acc  = result["best_val_acc"]
    sens = result.get("best_sens", 0.0)
    spec = result.get("best_spec", 0.0)
    print(f"\n{'='*60}")
    print(f"[{tag}] 最佳验证准确率:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"[{tag}] Micro Sensitivity: {sens:.4f} ({sens*100:.2f}%)")
    print(f"[{tag}] Micro Specificity: {spec:.4f} ({spec*100:.2f}%)")
    print(classification_report(result["best_labels"], result["best_preds"],
                                target_names=class_names))
    try:
        auc = roc_auc_score(result["best_labels"], result["best_probs"],
                             multi_class='ovr', average='macro')
        print(f"Macro AUC (OvR): {auc:.4f}")
        result["auc"] = auc
    except Exception as e:
        print(f"AUC 计算失败: {e}")


# =====================================================================
# 1. 线性探测评估
# =====================================================================
def run_linear_probe(pretrained_model: AIMv2OCTAnomalyDetector,
                     oct_root: Path = OCT_IMAGE_ROOT,
                     epochs:   int   = LINEAR_PROBE_EPOCHS) -> Dict:
    """
    线性探测：冻结编码器，只训练线性分类头。
    - 直接反映预训练表征的判别能力（无额外非线性）。
    - 对比基线：随机初始化编码器的线性探测准确率 ≈ 25%（随机猜测）。
    """
    print("\n" + "="*60)
    print("🔬 [Linear Probe] 线性探测分类评估")
    print("="*60)

    train_ds = LabeledOCTDataset(oct_root, split="train", augment=False)
    val_ds   = LabeledOCTDataset(oct_root, split="val",   augment=False)

    # [FIX-4] 原版 `if not train_ds` 对 Dataset 对象做布尔判断会报错或永为 True。
    #          改为显式 len() 检查。
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("❌ 标注数据集为空，跳过")
        return {}

    train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                               num_workers=4, pin_memory=True)

    probe = LinearProbeClassifier(pretrained_model).to(DEVICE)

    result = _train_classifier(probe, train_loader, val_loader,
                                epochs=epochs, lr=LINEAR_PROBE_LR,
                                tag="LinearProbe")
    _print_report(result, "LinearProbe")

    plot_accuracy_curves(result["train_accs"], result["val_accs"],
                         save_filename="linear_probe_accuracy.png")
    plot_confusion_matrix(result["best_labels"], result["best_preds"],
                          TRUE_CLASSES,
                          title=f"线性探测混淆矩阵（Acc={result['best_val_acc']:.4f}）",
                          save_filename="linear_probe_cm.png")
    if result.get("best_probs") is not None:
        plot_roc_curves(result["best_labels"], result["best_probs"],
                        TRUE_CLASSES,
                        title="线性探测ROC曲线（One-vs-Rest）",
                        save_filename="linear_probe_roc.png")
        plot_pr_curves(result["best_labels"], result["best_probs"],
                       TRUE_CLASSES,
                       title="线性探测PR曲线（精确率-召回率）",
                       save_filename="linear_probe_pr.png")
    return result




# =====================================================================
# 3. 少标签微调（1% / 10%）— 核心半监督评估实验
# =====================================================================
def run_finetune_few_shot(pretrained_model: AIMv2OCTAnomalyDetector,
                          oct_root: Path = OCT_IMAGE_ROOT,
                          epochs:   int  = FINETUNE_EPOCHS,
                          ratios:   list = None) -> Dict[str, Dict]:
    """
    少标签微调评估。

    实验设计：
      - 自监督预训练阶段：使用全量无标签训练集（仅 NORMAL）学习 backbone 表征
      - 微调阶段：仅使用训练集中 r% 的有标签样本（r ∈ {1, 10}）做监督微调
      - 评估阶段：在完整测试集（val）上报告 Accuracy / Sensitivity / Specificity

    采样策略：
      - 按类别分层采样，保证每个类别均有足够样本参与微调
      - 每类至少取 1 个样本（防止 min(n_per_class) = 0）

    保存指标：
      - best_val_acc  (Accuracy)
      - best_sens     (Micro Sensitivity，one-vs-rest)
      - best_spec     (Micro Specificity，one-vs-rest)

    参数：
      ratios: 标注比例列表，默认 [0.01, 0.10]
    """
    import random as _random

    if ratios is None:
        ratios = [0.01, 0.10]

    print("\n" + "="*60)
    print("🎯 [FewShot FineTune] 少标签微调评估")
    print(f"   标注比例: {[f'{int(r*100)}%' for r in ratios]}")
    print("="*60)

    full_train_ds = LabeledOCTDataset(oct_root, split="train", augment=True)
    val_ds        = LabeledOCTDataset(oct_root, split="val",   augment=False)

    if len(full_train_ds) == 0 or len(val_ds) == 0:
        print("❌ 标注数据集为空，跳过少标签微调")
        return {}

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ── 按类别建立索引，便于分层采样 ─────────────────────────────────
    from collections import defaultdict
    class_indices: Dict[int, list] = defaultdict(list)
    for idx, label in enumerate(full_train_ds.labels):
        class_indices[label].append(idx)

    all_results: Dict[str, Dict] = {}

    for ratio in ratios:
        tag = f"FewShot-{int(ratio*100)}%"
        print(f"\n  ── {tag} ──────────────────────────────────────────")

        # 分层采样：每类按 ratio 采样，至少 1 个
        sampled_indices = []
        for cls_idx, idx_list in sorted(class_indices.items()):
            n_sample = max(1, int(ratio * len(idx_list)))
            sampled  = _random.sample(idx_list, min(n_sample, len(idx_list)))
            sampled_indices.extend(sampled)
            cls_name = TRUE_CLASSES[cls_idx] if cls_idx < len(TRUE_CLASSES) else str(cls_idx)
            print(f"    类别 {cls_name}: 总 {len(idx_list)} → 采样 {len(sampled)}")

        print(f"    总采样量: {len(sampled_indices)} / {len(full_train_ds)} "
              f"({len(sampled_indices)/len(full_train_ds)*100:.2f}%)")

        subset       = Subset(full_train_ds, sampled_indices)
        train_loader = DataLoader(subset, batch_size=16, shuffle=True,
                                   num_workers=4, pin_memory=True,
                                   drop_last=False)

        # 每个 ratio 独立构建一个新的微调模型，避免权重污染
        ft_model = FineTuneClassifier(pretrained_model).to(DEVICE)

        # ── 少标签专用训练配置 ──────────────────────────────────────
        # 1% / 10% 数据量极少，需要更多 epoch 充分收敛，
        # 且头部学习率要比编码器层更大（分层lr在_train_classifier内处理）
        few_shot_epochs = max(epochs, 20)   # 至少20轮，给少量数据充分时间
        few_shot_lr     = FINETUNE_LR * 10  # 头部lr：5e-5→5e-4，编码器层×0.1

        result = _train_classifier(ft_model, train_loader, val_loader,
                                    epochs=few_shot_epochs,
                                    lr=few_shot_lr,
                                    tag=tag,
                                    patience=8)   # 更宽松的早停
        _print_report(result, tag)

        # ── 可视化 ──────────────────────────────────────────────────
        ratio_str = f"{int(ratio*100)}pct"
        plot_accuracy_curves(
            result["train_accs"], result["val_accs"],
            save_filename=f"finetune_fewshot_{ratio_str}_accuracy.png")
        plot_confusion_matrix(
            result["best_labels"], result["best_preds"],
            TRUE_CLASSES,
            title=(f"少标签微调{int(ratio*100)}%混淆矩阵"
                   f"（Acc={result['best_val_acc']:.4f}）"),
            save_filename=f"finetune_fewshot_{ratio_str}_cm.png")
        if result.get("best_probs") is not None:
            plot_roc_curves(
                result["best_labels"], result["best_probs"],
                TRUE_CLASSES,
                title=f"少标签微调{int(ratio*100)}% ROC曲线（One-vs-Rest）",
                save_filename=f"finetune_fewshot_{ratio_str}_roc.png")
            plot_pr_curves(
                result["best_labels"], result["best_probs"],
                TRUE_CLASSES,
                title=f"少标签微调{int(ratio*100)}% PR曲线（精确率-召回率）",
                save_filename=f"finetune_fewshot_{ratio_str}_pr.png")

        # ── 汇总打印 ─────────────────────────────────────────────────
        print(f"\n  📊 [{tag}] 完整测试集结果：")
        print(f"    Accuracy:    {result['best_val_acc']:.4f} "
              f"({result['best_val_acc']*100:.2f}%)")
        print(f"    Sensitivity: {result['best_sens']:.4f} "
              f"({result['best_sens']*100:.2f}%)")
        print(f"    Specificity: {result['best_spec']:.4f} "
              f"({result['best_spec']*100:.2f}%)")

        all_results[tag] = result

    # ── 跨比例对比汇总表 ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("📊 [FewShot FineTune] 跨比例汇总对比")
    print(f"  {'Ratio':<12} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13}")
    print("  " + "-"*50)
    for tag, res in all_results.items():
        print(f"  {tag:<12} "
              f"{res['best_val_acc']:>10.4f} "
              f"{res['best_sens']:>13.4f} "
              f"{res['best_spec']:>13.4f}")
    print("="*60)

    return all_results


# =====================================================================
# 4. 消融实验
# =====================================================================
def run_ablation(pretrained_model: AIMv2OCTAnomalyDetector,
                 oct_root: Path = OCT_IMAGE_ROOT) -> Dict:
    """
    消融实验：
      A. 减少标注数据量（1% / 5% / 10% / 20%），测试少样本鲁棒性
      B. （如有多模态模型）移除文本模态，对比精度差异
    """
    import random
    try:
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("⚠️  需要 scikit-learn")
        return {}

    print("\n" + "="*60)
    print("🔬 [Ablation] 消融实验 — 少样本鲁棒性")
    print("="*60)

    full_train = LabeledOCTDataset(oct_root, split="train")
    val_ds     = LabeledOCTDataset(oct_root, split="val")

    if len(full_train) == 0 or len(val_ds) == 0:
        print("❌ 数据集为空，跳过消融实验")
        return {}

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)

    ratios  = [0.01, 0.05, 0.10, 0.20, 1.0]
    results = {}

    for ratio in ratios:
        n       = max(1, int(ratio * len(full_train)))
        indices = random.sample(range(len(full_train)), n)
        subset  = Subset(full_train, indices)
        loader  = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4)

        probe = LinearProbeClassifier(pretrained_model).to(DEVICE)
        r     = _train_classifier(probe, loader, val_loader,
                                   epochs=5, lr=LINEAR_PROBE_LR,
                                   tag=f"Ablation-{int(ratio*100)}%")
        acc  = r["best_val_acc"]
        sens = r.get("best_sens", 0.0)
        spec = r.get("best_spec", 0.0)
        results[f"{int(ratio*100)}%"] = acc
        print(f"  标注量 {int(ratio*100)}% ({n} 张): "
              f"Acc={acc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

    # 保存消融结果图
    import matplotlib.pyplot as plt
    labels = list(results.keys())
    accs   = [results[k] for k in labels]
    plt.figure(figsize=(8, 4))
    plt.plot(labels, accs, 'bo-', linewidth=2)
    plt.xlabel("Training Label Ratio"); plt.ylabel("Val Accuracy")
    plt.title("Ablation: Few-shot Robustness"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PHOTO_SAVE_DIR / "ablation_few_shot.png", dpi=150)
    plt.close()
    print(f"✅ 消融图保存: {PHOTO_SAVE_DIR / 'ablation_few_shot.png'}")

    return results


# =====================================================================
# 5. 对比基线
# =====================================================================
def run_baseline_imagenet(oct_root: Path = OCT_IMAGE_ROOT) -> Dict:
    """
    ImageNet 迁移学习基线：
    直接使用未在 OCT 上预训练的 AIMv2 LiT 特征做线性探测。
    与自监督预训练结果对比，量化"OCT 域适配"的贡献。
    """
    print("\n" + "="*60)
    print("📊 [Baseline] ImageNet 直接迁移基线")
    print("="*60)

    baseline_model = AIMv2OCTAnomalyDetector()
    for p in baseline_model.parameters():
        p.requires_grad = False

    train_ds = LabeledOCTDataset(oct_root, split="train")
    val_ds   = LabeledOCTDataset(oct_root, split="val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("❌ 数据集为空，跳过基线实验")
        return {}

    train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    probe  = LinearProbeClassifier(baseline_model).to(DEVICE)
    result = _train_classifier(probe, train_loader, val_loader,
                                epochs=LINEAR_PROBE_EPOCHS,
                                lr=LINEAR_PROBE_LR,
                                tag="Baseline-ImageNet")
    _print_report(result, "Baseline-ImageNet")
    return result


def run_baseline_random(oct_root: Path = OCT_IMAGE_ROOT) -> Dict:
    """
    随机初始化基线：编码器权重随机，只训练线性头。
    提供性能下界（理论上接近 25% 随机猜测）。
    """
    print("\n" + "="*60)
    print("📊 [Baseline] 随机初始化基线")
    print("="*60)

    from aim.v2.torch.models import aimv2_large_lit

    class RandomEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = aimv2_large_lit()
            for p in self.enc.parameters():
                p.requires_grad = False

        def extract_patch_features(self, img_batch):
            B = img_batch.shape[0]
            from config import NUM_PATCHES, VISION_DIM
            return torch.randn(B, NUM_PATCHES, VISION_DIM, device=img_batch.device)

    class _RandomProbe(nn.Module):
        def __init__(self):
            super().__init__()
            from config import VISION_DIM, NUM_CLASSES
            self.enc  = RandomEncoder()
            self.head = nn.Linear(VISION_DIM, NUM_CLASSES)

        def forward(self, x):
            feat = self.enc.extract_patch_features(x).mean(1)
            return self.head(feat)

    train_ds = LabeledOCTDataset(oct_root, split="train")
    val_ds   = LabeledOCTDataset(oct_root, split="val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("❌ 数据集为空，跳过随机基线实验")
        return {}

    train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    probe  = _RandomProbe().to(DEVICE)
    result = _train_classifier(probe, train_loader, val_loader,
                                epochs=LINEAR_PROBE_EPOCHS,
                                lr=LINEAR_PROBE_LR,
                                tag="Baseline-Random")
    _print_report(result, "Baseline-Random")
    return result
