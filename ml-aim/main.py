"""
main.py — 完整实验入口

执行顺序（与研究计划严格对应）：
  Phase 1 │ 数据集加载 + 原始图像可视化
  Phase 2 │ Stage-1 纯视觉自回归预训练（AIMv2OCTAnomalyDetector）
  Phase 3 │ Stage-2 多模态自回归预训练（MultimodalAIMv2）[可选]
  Phase 4 │ 异常检测可视化
  Phase 5 │ 线性探测 + 有监督微调评估
  Phase 5b│ 少标签微调（1% / 10%）：完整测试集上的 Accuracy / Sensitivity / Specificity
  Phase 6 │ 消融实验 + 对比基线
  Phase 7 │ 汇总打印

运行方式：
    python main.py                   # 全流程（每次都重新训练）
    python main.py --skip-multimodal # 跳过多模态阶段（无 ClinicalBERT）
    python main.py --skip-baseline   # 跳过耗时的基线对比
"""
import sys
import random
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")

# ── 无界面后端须在首次 import pyplot 之前设置；中文字体与论文样式由 utils 统一配置
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "aimv1"))
sys.path.insert(0, str(PROJECT_ROOT / "aimv2"))

from utils    import (plot_original_images, visualize_anomaly_detection,
                      plot_loss_curves, plot_results_table)
from config   import (DEVICE, OCT_IMAGE_ROOT, PHOTO_SAVE_DIR, CKPT_SAVE_DIR,
                       TRAIN_EPOCHS, LINEAR_PROBE_EPOCHS, FINETUNE_EPOCHS,
                       USE_TEXT_MODALITY)
from datasets import PretrainDataset, MultimodalOCTDataset
from models   import AIMv2OCTAnomalyDetector, MultimodalAIMv2
from pretrain import Trainer
from evaluate import (run_linear_probe, run_finetune_few_shot,
                      run_ablation, run_baseline_imagenet)


# =====================================================================
# 主函数
# =====================================================================
def main(args):
    # ── 随机种子固定 ───────────────────────────────────────────────────
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 70)
    print("  AIMv2 视网膜 OCT 多模态自回归预训练 — 疾病分类研究")
    print("=" * 70)
    print(f"  设备:        {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  显存:        "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  OCT 数据:    {OCT_IMAGE_ROOT}")
    print(f"  输出目录:    {PHOTO_SAVE_DIR}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # Phase 1 — 数据集加载 & 原始图像可视化
    # ══════════════════════════════════════════════════════════════════
    print("\n📂 Phase 1: 加载数据集")
    train_dataset = PretrainDataset(OCT_IMAGE_ROOT, split="train")
    val_dataset   = PretrainDataset(OCT_IMAGE_ROOT, split="val")
    plot_original_images(val_dataset, save_filename="phase1_original_images.png",
                         seed=42)

    # ══════════════════════════════════════════════════════════════════
    # Phase 2 — Stage-1: 纯视觉自回归预训练（每次都重新训练）
    # ══════════════════════════════════════════════════════════════════
    print("\n🚀 Phase 2: Stage-1 纯视觉自回归预训练")
    vision_model = AIMv2OCTAnomalyDetector()
    trainer_s1   = Trainer(vision_model, train_dataset, val_dataset,
                            stage=1, epochs=TRAIN_EPOCHS)
    tl_s1, vl_s1 = trainer_s1.run()
    plot_loss_curves(tl_s1, vl_s1, save_filename="phase2_stage1_loss.png")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3 — Stage-2: 多模态自回归预训练（可选，每次都重新训练）
    # ══════════════════════════════════════════════════════════════════
    mm_model = None
    if USE_TEXT_MODALITY and not args.skip_multimodal:
        print("\n🚀 Phase 3: Stage-2 多模态预训练（图像 + 文本）")
        try:
            mm_train_ds = MultimodalOCTDataset(OCT_IMAGE_ROOT, split="train")
            mm_val_ds   = MultimodalOCTDataset(OCT_IMAGE_ROOT, split="val")
            mm_model    = MultimodalAIMv2(vision_model)
            trainer_s2  = Trainer(mm_model, mm_train_ds, mm_val_ds,
                                   stage=2, epochs=TRAIN_EPOCHS)
            tl_s2, vl_s2 = trainer_s2.run()
            plot_loss_curves(tl_s2, vl_s2, save_filename="phase3_stage2_loss.png")

        except Exception as e:
            print(f"⚠️  多模态预训练失败（已回退至纯视觉）: {e}")
            print("    常见原因: transformers 未安装 / ClinicalBERT 无法下载")
            mm_model = None
    else:
        print("\n⏭  Phase 3: 跳过多模态预训练（--skip-multimodal 或 USE_TEXT_MODALITY=False）")

    # ══════════════════════════════════════════════════════════════════
    # Phase 4 — 异常检测可视化
    # ══════════════════════════════════════════════════════════════════
    print("\n🖼️  Phase 4: 异常检测可视化")
    torch.cuda.empty_cache()
    visualize_anomaly_detection(vision_model, val_dataset,
                                  save_filename="phase4_anomaly_detection.png",
                                  seed=42)

    # ══════════════════════════════════════════════════════════════════
    # Phase 5 — 分类评估（线性探测 + 有监督微调）
    # ══════════════════════════════════════════════════════════════════
    print("\n📊 Phase 5: 分类评估")

    # 5a. 纯视觉预训练模型的线性探测（全量标签）
    lp_result = run_linear_probe(vision_model, OCT_IMAGE_ROOT,
                                  epochs=LINEAR_PROBE_EPOCHS)

    # 5b. 多模态模型的线性探测（若 Stage-2 成功）
    lp_mm_result = {}
    if mm_model is not None:
        print("\n  [多模态] 多模态编码器线性探测")
        lp_mm_result = run_linear_probe(
            mm_model.vision_model, OCT_IMAGE_ROOT,
            epochs=LINEAR_PROBE_EPOCHS)

    # ══════════════════════════════════════════════════════════════════
    # Phase 5b — 少标签微调（1% / 10%）——有监督微调唯一评估协议
    # 指标：Accuracy / Micro Sensitivity / Micro Specificity（one-vs-rest）
    # ══════════════════════════════════════════════════════════════════
    print("\n📊 Phase 5b: 少标签微调评估（1% / 10% 标注量）")
    few_shot_results = run_finetune_few_shot(
        vision_model,
        oct_root=OCT_IMAGE_ROOT,
        epochs=FINETUNE_EPOCHS,
        ratios=[0.01, 0.10]
    )

    # ══════════════════════════════════════════════════════════════════
    # Phase 6 — 消融实验 & 对比基线
    # ══════════════════════════════════════════════════════════════════
    print("\n🔬 Phase 6: 消融实验 & 对比基线")
    ablation_result = run_ablation(vision_model, OCT_IMAGE_ROOT)

    baseline_result = {}
    if not args.skip_baseline:
        baseline_result = run_baseline_imagenet(OCT_IMAGE_ROOT)
    else:
        print("⏭  跳过 ImageNet 基线（--skip-baseline）")

    # ══════════════════════════════════════════════════════════════════
    # Phase 7 — 最终汇总
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("🎉 完整实验流程完成！结果汇总")
    print("=" * 70)

    def _fmt(r, key="best_val_acc"):
        v = r.get(key, None)
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"  Stage-1 预训练最终 train loss : {tl_s1[-1]:.6f}")
    print(f"  Stage-1 预训练最终 val   loss : {vl_s1[-1]:.6f}")

    print(f"\n  {'实验':<30} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13}")
    print("  " + "-"*68)

    # 线性探测（全量）
    print(f"  {'线性探测 (纯视觉)':<30} "
          f"{_fmt(lp_result):>10} "
          f"{_fmt(lp_result, 'best_sens'):>13} "
          f"{_fmt(lp_result, 'best_spec'):>13}")

    # 线性探测（多模态）
    if lp_mm_result:
        print(f"  {'线性探测 (多模态)':<30} "
              f"{_fmt(lp_mm_result):>10} "
              f"{_fmt(lp_mm_result, 'best_sens'):>13} "
              f"{_fmt(lp_mm_result, 'best_spec'):>13}")

    # 少标签微调
    for tag, res in few_shot_results.items():
        label = f"少标签微调 ({tag})"
        print(f"  {label:<30} "
              f"{_fmt(res):>10} "
              f"{_fmt(res, 'best_sens'):>13} "
              f"{_fmt(res, 'best_spec'):>13}")

    # ImageNet 基线
    if baseline_result:
        print(f"  {'ImageNet 基线':<30} "
              f"{_fmt(baseline_result):>10} "
              f"{_fmt(baseline_result, 'best_sens'):>13} "
              f"{_fmt(baseline_result, 'best_spec'):>13}")

    # 消融实验
    if ablation_result:
        print("\n  消融实验（线性探测，少样本）:")
        for k, v in ablation_result.items():
            print(f"    {k:>6} 标注量: Acc = {v:.4f}")

    # ── 输出结果汇总图片 ──────────────────────────────────────────────
    table_rows = []
    table_rows.append(("线性探测 (纯视觉)",
                        lp_result.get("best_val_acc"),
                        lp_result.get("best_sens"),
                        lp_result.get("best_spec")))
    if lp_mm_result:
        table_rows.append(("线性探测 (多模态)",
                            lp_mm_result.get("best_val_acc"),
                            lp_mm_result.get("best_sens"),
                            lp_mm_result.get("best_spec")))
    for tag, res in few_shot_results.items():
        table_rows.append((f"少标签微调 ({tag})",
                            res.get("best_val_acc"),
                            res.get("best_sens"),
                            res.get("best_spec")))
    if baseline_result:
        table_rows.append(("ImageNet 基线",
                            baseline_result.get("best_val_acc"),
                            baseline_result.get("best_sens"),
                            baseline_result.get("best_spec")))
    plot_results_table(
        rows=table_rows,
        title="AIMv2 视网膜OCT疾病分类——实验结果汇总（Kaggle OCT2017）",
        save_filename="phase7_results_summary.png",
        extra_info={
            "Stage-1 训练损失": f"{tl_s1[-1]:.6f}",
            "Stage-1 验证损失": f"{vl_s1[-1]:.6f}",
        }
    )

    print(f"\n  输出目录: {PHOTO_SAVE_DIR}")
    print(f"  检查点:   {CKPT_SAVE_DIR}")
    print("=" * 70)


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    import multiprocessing
    if sys.platform != "linux":
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="AIMv2 OCT 多模态自回归实验")
    parser.add_argument("--skip-multimodal", action="store_true",
                        help="跳过 Stage-2 多模态预训练（无 HuggingFace 环境时使用）")
    parser.add_argument("--skip-baseline",   action="store_true",
                        help="跳过 ImageNet 基线对比（节省时间）")
    args = parser.parse_args()

    main(args)
