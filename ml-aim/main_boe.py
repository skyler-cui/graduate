"""
main_boe.py — BOE Publication Dataset 实验入口

[BOE-CHANGE] 本文件是 main.py 的替换版本，专用于 BOE 三分类数据集。
             原 main.py 保持不变，两套实验互不干扰。

与 main.py 的修改点汇总：
  [BOE-CHANGE-1]  从 config_boe 导入，而非 config
  [BOE-CHANGE-2]  从 datasets_boe 导入，而非 datasets
  [BOE-CHANGE-3]  evaluate.py 无需修改，但其内部 import 需指向 config_boe/datasets_boe
                  → 通过在本文件顶部 monkey-patch sys.modules 实现透明替换
  [BOE-CHANGE-4]  跳过多模态预训练（USE_TEXT_MODALITY=False）
  [BOE-CHANGE-5]  Phase 7 汇总标注数据集名称

运行方式：
    python main_boe.py                 # 全流程
    python main_boe.py --skip-baseline # 跳过 ImageNet 基线
"""
import sys
import random
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "aimv1"))
sys.path.insert(0, str(PROJECT_ROOT / "aimv2"))

# ── [BOE-CHANGE-3] Monkey-patch：让 evaluate.py / pretrain.py / models.py
#    在 import config / datasets 时自动拿到 BOE 版本，无需修改那些文件。
# ─────────────────────────────────────────────────────────────────────────────
import config_boe   as _cfg_boe
import datasets_boe as _ds_boe
sys.modules["config"]   = _cfg_boe   # evaluate/pretrain/models 中 `from config import ...` → BOE 版
sys.modules["datasets"] = _ds_boe    # evaluate 中 `from datasets import ...`  → BOE 版
# ─────────────────────────────────────────────────────────────────────────────

# [BOE-CHANGE-1] 从 config_boe 导入
from config_boe import (
    DEVICE, BOE_IMAGE_ROOT, PHOTO_SAVE_DIR, CKPT_SAVE_DIR,
    TRAIN_EPOCHS, LINEAR_PROBE_EPOCHS, FINETUNE_EPOCHS,
    USE_TEXT_MODALITY,
)

# [BOE-CHANGE-2] 从 datasets_boe 导入
from datasets_boe import PretrainDataset, MultimodalOCTDataset

from models   import AIMv2OCTAnomalyDetector, MultimodalAIMv2
from pretrain import Trainer
from evaluate import (run_linear_probe, run_finetune_few_shot,
                      run_ablation, run_baseline_imagenet)
from utils    import (plot_original_images, visualize_anomaly_detection,
                      plot_loss_curves, plot_results_table)


# =====================================================================
# 主函数
# =====================================================================
def main(args):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 70)
    # [BOE-CHANGE-5] 标注数据集名称
    print("  AIMv2 BOE 视网膜 OCT 三分类（AMD / DME / NORMAL）实验")
    print("=" * 70)
    print(f"  设备:        {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        print(f"  显存:        "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  BOE 数据:    {BOE_IMAGE_ROOT}")
    print(f"  输出目录:    {PHOTO_SAVE_DIR}")
    print(f"  类别:        AMD / DME / NORMAL（3 类）")
    print(f"  Train/Val:   83% / 17%（分层随机切分，seed=42）")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # Phase 1 — 数据集加载 & 原始图像可视化
    # ══════════════════════════════════════════════════════════════════
    print("\n📂 Phase 1: 加载数据集")
    train_dataset = PretrainDataset(BOE_IMAGE_ROOT, split="train")
    val_dataset   = PretrainDataset(BOE_IMAGE_ROOT, split="val")
    # 临时兼容：不传 seed（旧版 utils.py 也可运行）
    plot_original_images(val_dataset, save_filename="phase1_original_images_boe.png")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2 — Stage-1: 纯视觉自回归预训练
    # ══════════════════════════════════════════════════════════════════
    print("\n🚀 Phase 2: Stage-1 纯视觉自回归预训练")
    vision_model = AIMv2OCTAnomalyDetector()
    trainer_s1   = Trainer(vision_model, train_dataset, val_dataset,
                            stage=1, epochs=TRAIN_EPOCHS)
    tl_s1, vl_s1 = trainer_s1.run()
    plot_loss_curves(tl_s1, vl_s1, save_filename="phase2_stage1_loss_boe.png")

    # ══════════════════════════════════════════════════════════════════
    # [BOE-CHANGE-4] Phase 3 — 跳过多模态预训练（BOE 无文本配对）
    # ══════════════════════════════════════════════════════════════════
    mm_model = None
    print("\n⏭  Phase 3: 跳过多模态预训练（BOE 数据集无配对文本报告）")

    # ══════════════════════════════════════════════════════════════════
    # Phase 4 — 异常检测可视化
    # ══════════════════════════════════════════════════════════════════
    print("\n🖼️  Phase 4: 异常检测可视化")
    torch.cuda.empty_cache()
    # 临时兼容：不传 seed（旧版 utils.py 也可运行）
    visualize_anomaly_detection(
        vision_model,
        val_dataset,
        save_filename="phase4_anomaly_detection_boe.png"
    )

    # ══════════════════════════════════════════════════════════════════
    # Phase 5 — 分类评估（线性探测 + 有监督微调）
    # ══════════════════════════════════════════════════════════════════
    print("\n📊 Phase 5: 分类评估")

    lp_result = run_linear_probe(vision_model, BOE_IMAGE_ROOT,
                                  epochs=LINEAR_PROBE_EPOCHS)
    # ══════════════════════════════════════════════════════════════════
    # Phase 5b — 少标签微调（1% / 10%）
    # ══════════════════════════════════════════════════════════════════
    print("\n📊 Phase 5b: 少标签微调评估（1% / 10% 标注量）")
    few_shot_results = run_finetune_few_shot(
        vision_model,
        oct_root=BOE_IMAGE_ROOT,
        epochs=FINETUNE_EPOCHS,
        ratios=[0.01, 0.10]
    )

    # ══════════════════════════════════════════════════════════════════
    # Phase 6 — 消融实验 & 对比基线
    # ══════════════════════════════════════════════════════════════════
    print("\n🔬 Phase 6: 消融实验 & 对比基线")
    ablation_result = run_ablation(vision_model, BOE_IMAGE_ROOT)

    baseline_result = {}
    if not args.skip_baseline:
        baseline_result = run_baseline_imagenet(BOE_IMAGE_ROOT)
    else:
        print("⏭  跳过 ImageNet 基线（--skip-baseline）")

    # ══════════════════════════════════════════════════════════════════
    # Phase 7 — 最终汇总
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    # [BOE-CHANGE-5] 标注数据集名称
    print("🎉 BOE 实验完成！结果汇总（AMD / DME / NORMAL 三分类）")
    print("=" * 70)

    def _fmt(r, key="best_val_acc"):
        v = r.get(key, None)
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"  Stage-1 预训练最终 train loss : {tl_s1[-1]:.6f}")
    print(f"  Stage-1 预训练最终 val   loss : {vl_s1[-1]:.6f}")

    print(f"\n  {'实验':<30} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13}")
    print("  " + "-"*68)

    print(f"  {'线性探测 (纯视觉)':<30} "
          f"{_fmt(lp_result):>10} "
          f"{_fmt(lp_result, 'best_sens'):>13} "
          f"{_fmt(lp_result, 'best_spec'):>13}")

    for tag, res in few_shot_results.items():
        label = f"少标签微调 ({tag})"
        print(f"  {label:<30} "
              f"{_fmt(res):>10} "
              f"{_fmt(res, 'best_sens'):>13} "
              f"{_fmt(res, 'best_spec'):>13}")

    if baseline_result:
        print(f"  {'ImageNet 基线':<30} "
              f"{_fmt(baseline_result):>10} "
              f"{_fmt(baseline_result, 'best_sens'):>13} "
              f"{_fmt(baseline_result, 'best_spec'):>13}")

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
        title="AIMv2 视网膜OCT疾病分类——实验结果汇总（BOE三分类）",
        save_filename="phase7_results_summary_boe.png",
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

    parser = argparse.ArgumentParser(description="AIMv2 BOE 三分类实验")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 ImageNet 基线对比（节省时间）")
    args = parser.parse_args()

    main(args)
