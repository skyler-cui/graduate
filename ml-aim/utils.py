"""
utils.py — 工具函数

包含：
  - 图像反归一化
  - OCT ROI 掩膜生成
  - 可视化函数（原始图像、异常检测热力图、训练曲线）

字体规范（毕业论文）：
  - 中文字体：Noto Serif CJK（服务器已有，近似宋体）
  - 字号：小四号 = 12 pt
  - 图题、轴标签、刻度标签、图例均统一为 12 pt
  - 输出分辨率：300 DPI（满足印刷要求）
  - 线宽：2 pt（满足论文图线清晰度要求）
"""
import random
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import matplotlib.font_manager as _fm
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from scipy.ndimage import binary_opening, binary_closing

from config import (
    PHOTO_SAVE_DIR, VISUALIZE_NUM, DEVICE,
    ROI_THRESHOLD_LOW, ROI_THRESHOLD_HIGH,
    ROI_MORPHOLOGY_KERNEL, ABNORMAL_THRESHOLD_PERCENTILE,
    IMG_SIZE
)

# =====================================================================
# 全局字体与样式配置（毕业论文规范）
# 小四号 = 12 pt；宋体对应 SimSun（Windows/Linux 均可用）
# =====================================================================
_FONT_SIZE   = 12          # 小四号
_LINE_WIDTH  = 2.0         # 论文图线宽
_MARKER_SIZE = 6
_DPI         = 300         # 印刷分辨率


# ── 简体中文字体路径
# 从 NotoSerifCJK-Regular.ttc 提取的简体中文（SC）独立子集，
# 避免 .ttc 多语言集合文件被 matplotlib 错误映射到日文字形。
# 首次运行时自动提取并缓存到 /root/NotoSerifCJK-SC.ttf。
_SC_TTF_PATH = "/root/NotoSerifCJK-SC.ttf"
_TTC_SOURCE  = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
_SC_INDEX    = 2   # TTC 内简体中文子集索引（0=JP,1=KR,2=SC,3=TC,4=HK）


def _ensure_sc_font() -> str:
    """
    确保简体中文独立 .ttf 文件存在并已注册到 matplotlib。
    返回字体名称字符串供 rcParams 使用。

    原理：NotoSerifCJK-Regular.ttc 是多语言集合，matplotlib 默认只索引
    第一个子集（JP），中文字符因此显示为方框。
    解决方案：用 fonttools 一次性提取 SC 子集为独立 .ttf 并缓存，
    后续直接加载缓存文件，无需重复提取。
    """
    sc_path = Path(_SC_TTF_PATH)

    # 若缓存不存在则从 .ttc 提取
    if not sc_path.exists():
        try:
            from fontTools.ttLib import TTCollection
            tc = TTCollection(_TTC_SOURCE)
            tc[_SC_INDEX].save(str(sc_path))
            print(f"[utils] 已提取简体中文字体: {sc_path}")
        except Exception as e:
            print(f"[utils] 字体提取失败，中文可能显示异常: {e}")
            return "DejaVu Sans"

    # 注册到 matplotlib
    _fm.fontManager.addfont(str(sc_path))
    prop = _fm.FontProperties(fname=str(sc_path))
    return prop.get_name()   # "Noto Serif CJK SC"


def _apply_thesis_style():
    """
    统一设置 matplotlib 全局参数，使所有图符合毕业论文规范。
    模块加载时自动执行一次。
    """
    # ── 提取并注册简体中文字体
    sc_font_name = _ensure_sc_font()

    rcParams["font.family"]        = "sans-serif"
    rcParams["font.sans-serif"]    = [sc_font_name, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False      # 负号正常显示

    # ── 字号统一为小四号（12 pt）
    rcParams["font.size"]          = _FONT_SIZE
    rcParams["axes.titlesize"]     = _FONT_SIZE
    rcParams["axes.labelsize"]     = _FONT_SIZE
    rcParams["xtick.labelsize"]    = _FONT_SIZE
    rcParams["ytick.labelsize"]    = _FONT_SIZE
    rcParams["legend.fontsize"]    = _FONT_SIZE
    rcParams["figure.titlesize"]   = _FONT_SIZE

    # ── 线条与刻度
    rcParams["lines.linewidth"]    = _LINE_WIDTH
    rcParams["lines.markersize"]   = _MARKER_SIZE
    rcParams["axes.linewidth"]     = 1.0
    rcParams["xtick.major.width"]  = 1.0
    rcParams["ytick.major.width"]  = 1.0
    rcParams["xtick.direction"]    = "in"
    rcParams["ytick.direction"]    = "in"

    # ── 图例
    rcParams["legend.framealpha"]  = 0.9
    rcParams["legend.edgecolor"]   = "gray"

    # ── 保存参数
    rcParams["savefig.dpi"]        = _DPI
    rcParams["savefig.bbox"]       = "tight"
    rcParams["figure.dpi"]         = 100


_apply_thesis_style()


# =====================================================================
# 1. 图像反归一化
# =====================================================================
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    反 ImageNet 归一化，返回值域 [0, 1]。
    Args:
        tensor: [C, H, W] 或 [B, C, H, W]
    """
    mean = _MEAN.to(tensor.device)
    std  = _STD.to(tensor.device)

    if tensor.dim() == 4:
        mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    else:
        mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# =====================================================================
# 2. OCT ROI 掩膜
# =====================================================================
def create_roi_mask(img_tensor: torch.Tensor,
                    low:    float = ROI_THRESHOLD_LOW,
                    high:   float = ROI_THRESHOLD_HIGH,
                    kernel: int   = ROI_MORPHOLOGY_KERNEL) -> torch.Tensor:
    """
    生成 OCT 有效视网膜区域掩膜（排除纯黑/纯白背景）。
    支持单张 [C,H,W] 或批量 [B,C,H,W] 输入。
    返回: [H,W] 或 [B,H,W] bool Tensor
    """
    if img_tensor.dim() == 4:
        gray     = img_tensor.mean(dim=1)
        is_batch = True
    elif img_tensor.dim() == 3:
        gray     = img_tensor.mean(dim=0)
        is_batch = False
    else:
        gray     = img_tensor
        is_batch = False

    mask_np = ((gray > low) & (gray < high)).cpu().numpy()
    k = np.ones((kernel, kernel), dtype=bool)

    if is_batch:
        for i in range(mask_np.shape[0]):
            m = binary_opening(mask_np[i], structure=k)
            mask_np[i] = binary_closing(m, structure=k)
    else:
        mask_np = binary_closing(binary_opening(mask_np, structure=k), structure=k)

    return torch.from_numpy(mask_np).to(img_tensor.device)


def apply_roi_to_error(error_map: np.ndarray,
                       roi_mask:  torch.Tensor) -> np.ndarray:
    """将误差图与 ROI 掩膜对齐相乘，只保留有效区域误差。"""
    roi = roi_mask.squeeze()
    roi_resized = F.interpolate(
        roi.unsqueeze(0).unsqueeze(0).float(),
        size=error_map.shape,
        mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    return error_map * (roi_resized > 0.5).astype(float)


# =====================================================================
# 3. 可视化函数
# =====================================================================

def plot_original_images(dataset,
                         save_filename: str = "oct_original_images.png",
                         seed: Optional[int] = None):
    """
    绘制验证集原始灰度图与ROI掩膜叠加图。
    4行5列，图题及标注全部中文宋体小四号，300 DPI。
    """
    save_path = PHOTO_SAVE_DIR / save_filename
    n_pick    = min(VISUALIZE_NUM, len(dataset))
    if seed is not None:
        rng     = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_pick)
    else:
        indices = random.sample(range(len(dataset)), n_pick)
    subset    = Subset(dataset, indices)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.15)

    for i, ax in enumerate(axes.flatten()):
        if i >= len(subset):
            ax.axis("off")
            continue
        item = subset[i]
        t    = item[0] if isinstance(item, (list, tuple)) else item
        t    = t.to(DEVICE)

        rgb  = denormalize(t)
        gray = rgb.mean(0).cpu().numpy()
        roi  = create_roi_mask(rgb).cpu().numpy()

        ax.imshow(gray, cmap='gray')
        ax.imshow(roi, cmap='Reds', alpha=0.3)
        ax.set_title(f"图像 {i+1}\nROI掩膜叠加", fontsize=_FONT_SIZE)
        ax.axis("off")

    fig.suptitle("视网膜OCT图像——原始图像及ROI掩膜（验证集）",
                 fontsize=_FONT_SIZE, y=1.01)
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 原始图像保存: {save_path}")


def visualize_anomaly_detection(model,
                                dataset,
                                save_filename: str = "oct_anomaly_detection.png",
                                seed: Optional[int] = None):
    """
    异常检测可视化：每张图左列原始灰度图、右列重建误差热力图。

    阈值策略（双重门控）：
      1. global_thr 仅由异常类图像（非NORMAL）的有效误差分布决定，
         取第 ABNORMAL_THRESHOLD_PERCENTILE 百分位，与正常图误差完全解耦。
      2. 仅对 is_normal=False 的图像才绘制标注框，
         且要求超过阈值的区域占有效ROI像素比例 >= MIN_ABNORMAL_RATIO（2%），
         过滤面积极小的噪声性误差峰值，防止假阳性标注。
    正常图（NORMAL）绝对不绘制任何标注框，热力图颜色范围独立归一化。
    """
    from datasets import parse_true_class

    overlay_cmap = LinearSegmentedColormap.from_list(
        "anomaly_overlay",
        [(0.00, (0.00, 0.00, 1.0, 0.00)),
         (0.40, (0.00, 0.00, 1.0, 0.00)),
         (0.60, (1.00, 1.00, 0.0, 0.50)),
         (0.80, (1.00, 0.50, 0.0, 0.70)),
         (1.00, (1.00, 0.00, 0.0, 0.90)),
         ]
    )

    CLS_COLOR = {
        "CNV":    "#D62728",
        "DME":    "#FF7F0E",
        "DRUSEN": "#BCBD22",
        "AMD":    "#D62728",
        "NORMAL": "#2CA02C",
    }
    CLS_LABEL_CN = {
        "CNV":    "CNV（脉络膜新生血管）",
        "DME":    "DME（糖尿病黄斑水肿）",
        "DRUSEN": "DRUSEN（玻璃膜疣）",
        "AMD":    "AMD（年龄相关性黄斑变性）",
        "NORMAL": "NORMAL（正常视网膜）",
    }
    MIN_ABNORMAL_RATIO = 0.02   # 超阈值区域至少占有效ROI的2%才画框

    save_path = PHOTO_SAVE_DIR / save_filename
    n_pick    = min(VISUALIZE_NUM, len(dataset))
    if seed is not None:
        rng     = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n_pick)
    else:
        indices = random.sample(range(len(dataset)), n_pick)
    subset    = Subset(dataset, indices)
    model.eval()

    # ── 第一遍：提取所有误差图，分别收集正常/异常的有效误差 ──────────
    records             = []
    normal_valid_errs   = []
    abnormal_valid_errs = []

    for idx in range(len(subset)):
        item = subset[idx]
        t    = (item[0] if isinstance(item, (list, tuple)) else item).unsqueeze(0).to(DEVICE)

        raw_idx    = indices[idx]
        img_paths_ = getattr(dataset, "img_paths", [None] * len(dataset))
        img_path   = img_paths_[raw_idx] if img_paths_ else None
        cls_name   = parse_true_class(img_path) if img_path else "UNKNOWN"
        is_normal  = (cls_name == "NORMAL")

        rgb      = denormalize(t)
        roi_mask = create_roi_mask(rgb)
        gray_np  = rgb.squeeze(0).mean(0).cpu().numpy()

        with torch.no_grad():
            pred = model.autoregressive_predict(t)
            true = model.extract_patch_features(t)
            err  = F.mse_loss(pred, true, reduction="none").mean(-1).cpu()

        n_side   = int(err.shape[1] ** 0.5)
        err_full = F.interpolate(
            err.reshape(1, 1, n_side, n_side).float(),
            size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        roi_np     = roi_mask.squeeze(0).cpu().numpy()
        masked_err = err_full * roi_np.astype(float)

        records.append({
            "gray": gray_np, "masked_err": masked_err,
            "roi_np": roi_np, "cls": cls_name, "is_normal": is_normal,
        })

        valid_px = masked_err[masked_err > 0]
        if len(valid_px):
            (normal_valid_errs if is_normal else abnormal_valid_errs).append(valid_px)

    # ── 计算阈值（仅来自异常图）与各自 vmax ──────────────────────────
    if abnormal_valid_errs:
        abn_flat   = np.concatenate(abnormal_valid_errs)
        global_thr = np.percentile(abn_flat, ABNORMAL_THRESHOLD_PERCENTILE)
        abn_vmax   = np.percentile(abn_flat, 97)
    else:
        global_thr = abn_vmax = 0.0

    if normal_valid_errs:
        norm_vmax = np.percentile(np.concatenate(normal_valid_errs), 97)
    else:
        norm_vmax = abn_vmax

    print(f"  [AnomalyViz] global_thr={global_thr:.6f}  "
          f"abn_vmax={abn_vmax:.6f}  norm_vmax={norm_vmax:.6f}")

    # ── 布局：每行 PAIRS_PER_ROW 对，每对2列（左=灰度，右=热力图）──
    PAIRS_PER_ROW = 4
    n_imgs    = len(records)
    n_rows    = int(np.ceil(n_imgs / PAIRS_PER_ROW))
    n_subcols = PAIRS_PER_ROW * 2

    fig, axes = plt.subplots(
        n_rows, n_subcols,
        figsize=(n_subcols * 2.0, n_rows * 2.7),
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.05)

    for idx, rec in enumerate(records):
        row      = idx // PAIRS_PER_ROW
        pc       = idx %  PAIRS_PER_ROW
        ax_gray  = axes[row][pc * 2]
        ax_heat  = axes[row][pc * 2 + 1]

        gray       = rec["gray"]
        masked_err = rec["masked_err"]
        roi_np     = rec["roi_np"]
        cls_name   = rec["cls"] or "UNKNOWN"
        is_normal  = rec["is_normal"]
        color      = CLS_COLOR.get(cls_name, "#888888")
        vmax       = norm_vmax if is_normal else abn_vmax

        # ── 预计算异常标注框坐标（供左右两图共用）──────────────────
        bbox_coords = None   # (x0, y0, x1, y1) or None
        if not is_normal and global_thr > 0:
            abnormal_mask = masked_err > global_thr
            roi_total     = roi_np.sum()
            abn_ratio     = abnormal_mask.sum() / (roi_total + 1e-8)
            if abnormal_mask.any() and abn_ratio >= MIN_ABNORMAL_RATIO:
                ys, xs = np.where(abnormal_mask)
                bbox_coords = (int(xs.min()), int(ys.min()),
                               int(xs.max()), int(ys.max()))

        def _draw_bbox(ax, dashed=False):
            """在指定 axes 上绘制标注框 + 标签。"""
            x0, y0, x1, y1 = bbox_coords
            ls = (0, (4, 2)) if dashed else "solid"   # 原图用虚线，热力图用实线
            rect = mpatches.FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0,
                boxstyle="round,pad=2",
                linewidth=1.6, edgecolor=color,
                linestyle=ls,
                facecolor="none", zorder=5
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 2, max(y0 - 4, 4),
                CLS_LABEL_CN.get(cls_name, cls_name),
                fontsize=7, color="white", weight="bold", va="bottom",
                bbox=dict(facecolor=color, alpha=0.8, pad=1.0,
                          edgecolor="none", boxstyle="round,pad=0.3")
            )

        # ── 左图：原始灰度 + 异常标注框（虚线） ───────────────────
        ax_gray.imshow(gray, cmap="gray", vmin=0, vmax=1)
        if bbox_coords is not None:
            _draw_bbox(ax_gray, dashed=True)
        status_tag = "✓ 正常" if is_normal else "⚠ 异常"
        ax_gray.set_title(
            f"图{idx+1}  {cls_name}\n{status_tag}",
            fontsize=_FONT_SIZE - 1, color=color, pad=2
        )
        ax_gray.axis("off")

        # ── 右图：热力图叠加 + 异常标注框（实线）─────────────────
        ax_heat.imshow(gray, cmap="gray", vmin=0, vmax=1)
        err_show = masked_err.copy()
        valid_px = masked_err[masked_err > 0]
        if len(valid_px):
            err_show[err_show < np.percentile(valid_px, 50)] = 0
        ax_heat.imshow(err_show, cmap=overlay_cmap,
                       vmin=0, vmax=vmax, interpolation="bilinear", alpha=0.9)
        if bbox_coords is not None:
            _draw_bbox(ax_heat, dashed=False)

        heat_title = (
            f"重建误差热力图\nthr={global_thr:.2e}" if not is_normal
            else "重建误差热力图\n（正常，无标注框）"
        )
        ax_heat.set_title(heat_title, fontsize=_FONT_SIZE - 1, color=color, pad=2)
        ax_heat.axis("off")

    # 关闭多余子图
    for idx in range(n_imgs, n_rows * PAIRS_PER_ROW):
        row = idx // PAIRS_PER_ROW
        pc  = idx %  PAIRS_PER_ROW
        axes[row][pc * 2].axis("off")
        axes[row][pc * 2 + 1].axis("off")

    legend_handles = [
        mpatches.Patch(color=CLS_COLOR[c], label=CLS_LABEL_CN[c])
        for c in CLS_LABEL_CN
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=min(len(legend_handles), 5),
        fontsize=_FONT_SIZE - 1, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        f"AIMv2视网膜OCT异常检测  ——  重建误差热力图与异常区域标注\n"
        f"（标注框阈值: {global_thr:.4e}，仅对异常图标注，正常图绝对无边框）",
        fontsize=_FONT_SIZE, y=1.01
    )
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 异常检测结果保存: {save_path}")

def plot_loss_curves(train_losses, val_losses=None,
                     save_filename: str = "training_loss.png"):
    """
    绘制训练/验证损失曲线。
    中文标签、宋体小四号、刻度朝内、300 DPI。
    """
    save_path = PHOTO_SAVE_DIR / save_filename
    epochs    = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses,
            'b-o', linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
            label='训练损失')
    if val_losses:
        ax.plot(list(range(1, len(val_losses) + 1)), val_losses,
                'r-s', linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
                label='验证损失')

    for i, v in enumerate(train_losses):
        ax.annotate(f'{v:.5f}', (i + 1, v),
                    textcoords="offset points", xytext=(0, 6),
                    ha='center', fontsize=10)

    ax.set_xlabel('轮次（Epoch）', fontsize=_FONT_SIZE)
    ax.set_ylabel('均方误差损失（MSE Loss）', fontsize=_FONT_SIZE)
    ax.set_title('AIMv2视网膜OCT自监督预训练损失曲线', fontsize=_FONT_SIZE)
    ax.set_xticks(epochs)
    ax.legend(fontsize=_FONT_SIZE)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=_FONT_SIZE, direction='in')

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失曲线保存: {save_path}")


def plot_ablation_label_ratio_curves(ratio_labels: list,
                                     accuracies: list,
                                     sensitivities: Optional[list] = None,
                                     specificities: Optional[list] = None,
                                     save_filename: str = "ablation_few_shot.png",
                                     title: str = "消融实验：训练标注比例与验证指标（线性探测）"):
    """
    消融曲线（少标签比例 vs 指标）出图。
    与本项目论文图风格一致（中文标签、小四号、300 DPI）。
    """
    save_path = PHOTO_SAVE_DIR / save_filename

    if len(ratio_labels) == 0:
        print("⚠️  消融曲线：ratio_labels 为空，跳过绘图")
        return

    x = np.arange(len(ratio_labels))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, accuracies,
            'o-', color="#1F77B4",
            linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
            label="Accuracy")

    if sensitivities is not None and len(sensitivities) == len(ratio_labels):
        ax.plot(x, sensitivities,
                's-', color="#D62728",
                linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
                label="Sensitivity（micro）")

    if specificities is not None and len(specificities) == len(ratio_labels):
        ax.plot(x, specificities,
                '^-', color="#2CA02C",
                linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
                label="Specificity（micro）")

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels, fontsize=_FONT_SIZE)
    ax.set_xlabel("训练集标注比例", fontsize=_FONT_SIZE)
    ax.set_ylabel("指标值（0–1）", fontsize=_FONT_SIZE)
    ax.set_title(title, fontsize=_FONT_SIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=_FONT_SIZE, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=_FONT_SIZE, direction="in")

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 消融曲线保存: {save_path}")


# =====================================================================
# 准确率曲线
# =====================================================================
def plot_accuracy_curves(train_accs, val_accs,
                         save_filename: str = "accuracy_curve.png"):
    """
    绘制分类准确率曲线（训练集 + 验证集）。
    Y轴为百分比，标注最优验证准确率，300 DPI。
    """
    save_path = PHOTO_SAVE_DIR / save_filename
    epochs    = list(range(1, len(train_accs) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, [a * 100 for a in train_accs],
            'b-o', linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
            label='训练准确率')
    ax.plot(epochs, [a * 100 for a in val_accs],
            'r-s', linewidth=_LINE_WIDTH, markersize=_MARKER_SIZE,
            label='验证准确率')

    best_ep  = int(np.argmax(val_accs))
    best_val = val_accs[best_ep] * 100
    ax.annotate(f'最优：{best_val:.2f}%',
                xy=(best_ep + 1, best_val),
                xytext=(best_ep + 1, min(best_val + 5, 103)),
                ha='center', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax.set_xlabel('轮次（Epoch）', fontsize=_FONT_SIZE)
    ax.set_ylabel('准确率（%）', fontsize=_FONT_SIZE)
    ax.set_title('分类准确率曲线', fontsize=_FONT_SIZE)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=_FONT_SIZE)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=_FONT_SIZE, direction='in')

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 准确率曲线保存: {save_path}")


# =====================================================================
# 混淆矩阵
# =====================================================================
def plot_confusion_matrix(labels, preds, class_names,
                          title: str = "混淆矩阵",
                          save_filename: str = "confusion_matrix.png"):
    """
    绘制混淆矩阵热力图。
    每格同时显示样本计数与行归一化百分比，满足论文规范。
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("⚠️  绘制混淆矩阵需要 scikit-learn: pip install scikit-learn")
        return

    save_path = PHOTO_SAVE_DIR / save_filename
    cm        = confusion_matrix(labels, preds)
    n_cls     = cm.shape[0]

    # 行归一化（召回率视角）
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, n_cls * 1.8), max(5, n_cls * 1.5)))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('归一化比例', fontsize=_FONT_SIZE)
    cbar.ax.tick_params(labelsize=_FONT_SIZE)

    ticks = np.arange(n_cls)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=_FONT_SIZE, rotation=30, ha='right')
    ax.set_yticklabels(class_names, fontsize=_FONT_SIZE)

    thresh = 0.5
    for i in range(n_cls):
        for j in range(n_cls):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=_FONT_SIZE - 1, color=color)

    ax.set_ylabel('真实标签', fontsize=_FONT_SIZE)
    ax.set_xlabel('预测标签', fontsize=_FONT_SIZE)
    ax.set_title(title, fontsize=_FONT_SIZE, pad=10)
    ax.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵保存: {save_path}")

# =====================================================================
# ROC 曲线（One-vs-Rest，每类一条 + micro/macro 平均）
# =====================================================================
def _smooth_curve(x, y, n_points=300):
    """
    对曲线做单调插值平滑，消除阈值锯齿。
    x 必须单调，y 在 x 定义域内插值。
    返回等间距的平滑 (xs, ys)。
    """
    from scipy.interpolate import PchipInterpolator
    # 去重并排序
    order = np.argsort(x)
    xu, idx = np.unique(x[order], return_index=True)
    yu = y[order][idx]
    if len(xu) < 2:
        return x, y
    interp = PchipInterpolator(xu, yu)
    xs = np.linspace(xu[0], xu[-1], n_points)
    ys = np.clip(interp(xs), 0.0, 1.0)
    return xs, ys


def plot_roc_curves(labels, probs, class_names,
                    title: str = "ROC曲线",
                    save_filename: str = "roc_curve.png"):
    """
    绘制多分类 ROC 曲线（One-vs-Rest 策略）。

    修复：
      - 所有曲线使用 PchipInterpolator 平滑，消除阈值锯齿
      - 全部使用实线，通过颜色区分各类及平均线，无虚线
      - Micro/Macro 平均使用深灰、黑色实线，线宽略粗以突出

    参数：
      labels:      真实标签列表，shape (N,)，整数
      probs:       softmax 概率矩阵，shape (N, C)
      class_names: 类别名称列表，len = C
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    save_path = PHOTO_SAVE_DIR / save_filename
    labels_np = np.array(labels)
    probs_np  = np.array(probs)
    n_cls     = len(class_names)
    classes   = list(range(n_cls))

    labels_bin = label_binarize(labels_np, classes=classes)

    # ── 每类 ROC ──────────────────────────────────────────────────────
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ── Micro-average ─────────────────────────────────────────────────
    fpr["micro"], tpr["micro"], _ = roc_curve(
        labels_bin.ravel(), probs_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # ── Macro-average（统一 FPR 轴插值平均）──────────────────────────
    all_fpr  = np.linspace(0, 1, 300)
    mean_tpr = np.zeros(300)
    for i in range(n_cls):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_cls
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # ── 颜色方案：各类用明亮色，平均线用黑/深灰 ─────────────────────
    _COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E",
               "#9467BD", "#8C564B", "#E377C2", "#17BECF"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # 各类曲线——实线，平滑处理
    for i, cname in enumerate(class_names):
        sx, sy = _smooth_curve(fpr[i], tpr[i])
        ax.plot(sx, sy,
                color=_COLORS[i % len(_COLORS)],
                linewidth=_LINE_WIDTH,
                linestyle="-",
                label=f"{cname}（AUC = {roc_auc[i]:.3f}）")

    # Micro 平均——黑色实线，略粗
    sx, sy = _smooth_curve(fpr["micro"], tpr["micro"])
    ax.plot(sx, sy,
            color="black", linewidth=_LINE_WIDTH + 0.5,
            linestyle="-",
            label=f"Micro均值（AUC = {roc_auc['micro']:.3f}）")

    # Macro 平均——深灰实线
    ax.plot(fpr["macro"], tpr["macro"],
            color="#555555", linewidth=_LINE_WIDTH + 0.5,
            linestyle="-",
            label=f"Macro均值（AUC = {roc_auc['macro']:.3f}）")

    # 随机猜测基准——浅灰细实线
    ax.plot([0, 1], [0, 1],
            color="#BBBBBB", linewidth=1.0,
            linestyle="-", label="随机猜测基准")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("假阳性率 (FPR)", fontsize=_FONT_SIZE)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=_FONT_SIZE)
    ax.set_title(title, fontsize=_FONT_SIZE)
    ax.legend(loc="lower right", fontsize=_FONT_SIZE - 1,
              framealpha=0.9, edgecolor="gray")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_SIZE, direction="in")

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ ROC曲线保存: {save_path}")
    return roc_auc


# =====================================================================
# PR 曲线（One-vs-Rest，每类一条 + micro/macro 平均）
# =====================================================================
def plot_pr_curves(labels, probs, class_names,
                   title: str = "PR曲线（精确率-召回率）",
                   save_filename: str = "pr_curve.png"):
    """
    绘制多分类 PR 曲线（One-vs-Rest 策略）。

    修复：
      - PR 曲线原始形状为阶梯形，用统一 recall 轴插值后平滑，消除锯齿
      - 全部使用实线，无虚线
      - Micro/Macro 平均用黑/深灰实线，线宽略粗

    参数同 plot_roc_curves。
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    save_path = PHOTO_SAVE_DIR / save_filename
    labels_np = np.array(labels)
    probs_np  = np.array(probs)
    n_cls     = len(class_names)
    classes   = list(range(n_cls))

    labels_bin = label_binarize(labels_np, classes=classes)

    # ── 统一 recall 轴，用于平滑插值 ────────────────────────────────
    _RECALL_GRID = np.linspace(0, 1, 300)

    # ── 每类 PR ───────────────────────────────────────────────────────
    precision, recall, ap = {}, {}, {}
    prec_smooth = {}
    for i in range(n_cls):
        p, r, _ = precision_recall_curve(labels_bin[:, i], probs_np[:, i])
        ap[i]   = average_precision_score(labels_bin[:, i], probs_np[:, i])
        # PR 曲线 r 从大到小，翻转后插值
        prec_smooth[i] = np.clip(
            np.interp(_RECALL_GRID, r[::-1], p[::-1]), 0.0, 1.0)
        precision[i], recall[i] = p, r

    # ── Micro-average ─────────────────────────────────────────────────
    p_micro, r_micro, _ = precision_recall_curve(
        labels_bin.ravel(), probs_np.ravel())
    ap["micro"] = average_precision_score(labels_bin, probs_np, average="micro")
    prec_smooth["micro"] = np.clip(
        np.interp(_RECALL_GRID, r_micro[::-1], p_micro[::-1]), 0.0, 1.0)

    # ── Macro-average ─────────────────────────────────────────────────
    mean_prec = np.mean([prec_smooth[i] for i in range(n_cls)], axis=0)
    ap["macro"] = average_precision_score(labels_bin, probs_np, average="macro")

    # ── 颜色方案 ─────────────────────────────────────────────────────
    _COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E",
               "#9467BD", "#8C564B", "#E377C2", "#17BECF"]

    baseline = labels_bin.mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    # 各类曲线——实线，平滑
    for i, cname in enumerate(class_names):
        ax.plot(_RECALL_GRID, prec_smooth[i],
                color=_COLORS[i % len(_COLORS)],
                linewidth=_LINE_WIDTH,
                linestyle="-",
                label=f"{cname}（AP = {ap[i]:.3f}）")

    # Micro 平均——黑色实线
    ax.plot(_RECALL_GRID, prec_smooth["micro"],
            color="black", linewidth=_LINE_WIDTH + 0.5,
            linestyle="-",
            label=f"Micro均值（AP = {ap['micro']:.3f}）")

    # Macro 平均——深灰实线
    ax.plot(_RECALL_GRID, mean_prec,
            color="#555555", linewidth=_LINE_WIDTH + 0.5,
            linestyle="-",
            label=f"Macro均值（AP = {ap['macro']:.3f}）")

    # 随机猜测基准——浅灰细实线
    ax.axhline(y=baseline, color="#BBBBBB", linewidth=1.0,
               linestyle="-", label=f"随机猜测基准 ({baseline:.2f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("召回率 (Recall)", fontsize=_FONT_SIZE)
    ax.set_ylabel("精确率 (Precision)", fontsize=_FONT_SIZE)
    ax.set_title(title, fontsize=_FONT_SIZE)
    ax.legend(loc="lower left", fontsize=_FONT_SIZE - 1,
              framealpha=0.9, edgecolor="gray")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=_FONT_SIZE, direction="in")

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ PR曲线保存: {save_path}")
    return ap


# =====================================================================
# 实验结果汇总表图片
# =====================================================================
def plot_results_table(rows: list,
                       title: str = "实验结果汇总",
                       save_filename: str = "results_summary.png",
                       extra_info: dict = None):
    """
    将最终分类结果汇总渲染为图片。
    使用手动绘制方式（Rectangle + Text）彻底避免 ax.table 的重叠问题。

    参数：
      rows:       列表，每项为 (实验名, Accuracy, Sensitivity, Specificity)
      title:      图标题
      save_filename: 保存文件名
      extra_info: 额外信息字典，显示在表格下方
    """
    save_path = PHOTO_SAVE_DIR / save_filename

    # ── 布局参数 ─────────────────────────────────────────────────────
    COL_HEADERS  = ["实验配置", "Accuracy", "Sensitivity", "Specificity"]
    COL_WIDTHS   = [0.38, 0.20, 0.21, 0.21]   # 各列宽度比例（总和=1）
    ROW_H        = 0.13                         # 每行高度（axes坐标）
    HEADER_H     = 0.15                         # 表头行高
    LEFT         = 0.0                          # 表格左边界
    TOP          = 0.97                         # 表格顶边界

    n_rows  = len(rows)
    n_cols  = len(COL_HEADERS)
    total_h = HEADER_H + n_rows * ROW_H + 0.05  # 表格总高度

    # 图片高度随行数动态增长
    extra_h = 0.08 if extra_info else 0.0
    fig_h   = max(3.5, total_h * 9 + 1.2 + extra_h * 9)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 颜色 ─────────────────────────────────────────────────────────
    HEADER_BG    = "#2C5F8A"
    ROW_BG       = ["#EBF3FB", "#FFFFFF"]
    BEST_BG      = "#D6F0E0"
    BEST_EDGE    = "#1A7A3C"
    BEST_FC      = "#1A7A3C"
    GRID_COLOR   = "#AAAAAA"

    # ── 找各指标最优行 ────────────────────────────────────────────────
    def _best(col):
        vals = [r[col] if isinstance(r[col], float) else -1.0 for r in rows]
        return int(np.argmax(vals))

    best = {1: _best(1), 2: _best(2), 3: _best(3)}

    # ── 计算各列左边界 x ──────────────────────────────────────────────
    col_x = [LEFT]
    for w in COL_WIDTHS[:-1]:
        col_x.append(col_x[-1] + w)

    # ── 缩放到 axes 坐标：表格占 axes 的 [0,1]×[1-total_h, 1] ───────
    # 为了让表格居中且有标题空间，做一个简单的线性映射
    scale_y = (TOP - 0.03) / total_h   # 每单位高度对应 axes y 比例

    def row_top(i):
        """第 i 行（0=表头）的顶部 y 坐标（axes）"""
        if i == 0:
            return TOP
        return TOP - HEADER_H * scale_y - (i - 1) * ROW_H * scale_y

    def row_bot(i):
        h = HEADER_H if i == 0 else ROW_H
        return row_top(i) - h * scale_y

    # ── 绘制表头 ─────────────────────────────────────────────────────
    for j, (label, x, w) in enumerate(zip(COL_HEADERS, col_x, COL_WIDTHS)):
        y_top = row_top(0)
        y_bot = row_bot(0)
        rect = plt.Rectangle((x, y_bot), w, y_top - y_bot,
                              facecolor=HEADER_BG, edgecolor=GRID_COLOR,
                              linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, (y_top + y_bot) / 2, label,
                ha="center", va="center",
                color="white", fontweight="bold",
                fontsize=_FONT_SIZE, zorder=3)

    # ── 绘制数据行 ────────────────────────────────────────────────────
    def _fmt_cell(v):
        return "N/A" if v is None else f"{v:.4f}  ({v*100:.2f}%)"

    for i, (name, acc, sens, spec) in enumerate(rows):
        vals    = [name, acc, sens, spec]
        y_top   = row_top(i + 1)
        y_bot   = row_bot(i + 1)
        mid_y   = (y_top + y_bot) / 2
        bg      = ROW_BG[i % 2]

        for j, (x, w) in enumerate(zip(col_x, COL_WIDTHS)):
            is_best = (j in best and best[j] == i)
            face    = BEST_BG if is_best else bg
            edge    = BEST_EDGE if is_best else GRID_COLOR
            lw      = 1.2 if is_best else 0.8

            rect = plt.Rectangle((x, y_bot), w, y_top - y_bot,
                                  facecolor=face, edgecolor=edge,
                                  linewidth=lw, zorder=2)
            ax.add_patch(rect)

            # 文字内容
            if j == 0:
                txt   = name
                fc    = "#1A1A1A"
                fw    = "normal"
                fsize = _FONT_SIZE - 1
                ha    = "center"
            else:
                v     = vals[j]
                txt   = _fmt_cell(v)
                fc    = BEST_FC if is_best else "#1A1A1A"
                fw    = "bold" if is_best else "normal"
                fsize = _FONT_SIZE - 1
                ha    = "center"

            ax.text(x + w / 2, mid_y, txt,
                    ha=ha, va="center",
                    color=fc, fontweight=fw,
                    fontsize=fsize, zorder=3)

    # ── 标题 ─────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=_FONT_SIZE, pad=10, fontweight="bold")

    # ── 底部说明 ─────────────────────────────────────────────────────
    footer_y = row_bot(n_rows) - 0.04
    ax.text(0.5, footer_y,
            "★ 绿色背景为各指标最优结果",
            ha="center", va="top",
            fontsize=_FONT_SIZE - 2, color=BEST_FC)

    if extra_info:
        info_str = "   |   ".join(f"{k}: {v}" for k, v in extra_info.items())
        ax.text(0.5, footer_y - 0.05,
                info_str,
                ha="center", va="top",
                fontsize=_FONT_SIZE - 2, color="#555555")

    fig.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"✅ 结果汇总表保存: {save_path}")