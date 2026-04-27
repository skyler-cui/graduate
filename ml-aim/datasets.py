"""
datasets.py — 数据集定义

数据集结构（实际情况）：
┌─ oct_images/
│   ├─ train/
│   │   ├─ AMD/        ← ⚠️ 容器目录，内含 CNV/DME/DRUSEN 混合图像
│   │   │              ← 文件名前缀决定真实类别：CNV-1569-1.tif → CNV
│   │   ├─ CNV/        ← 纯 CNV 图像
│   │   ├─ DME/        ← 纯 DME 图像
│   │   ├─ DRUSEN/     ← 纯 DRUSEN 图像
│   │   └─ NORMAL/     ← 正常图像（自监督预训练唯一数据源）
│   └─ val/            ← 同上结构

真实分类任务：4类（CNV / DME / DRUSEN / NORMAL）
  - AMD 目录内图像通过文件名前缀归入 CNV/DME/DRUSEN 三类
  - AMD 目录本身不是独立类别

三类 Dataset：
  1. PretrainDataset       — 无标签，仅 NORMAL，自监督重建预训练
  2. LabeledOCTDataset     — 带标签，含 AMD 目录解析，分类评估/微调
  3. MultimodalOCTDataset  — 无标签 + 文本，多模态预训练（可选）
"""

import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import (
    OCT_IMAGE_ROOT, IMG_SIZE,
    MAX_TEXT_LEN, TEXT_MODEL_NAME
)

# ── 全局类别定义 ──────────────────────────────────────────────────────────────
# AMD 不是独立类别，只是一个容器目录
TRUE_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRUE_CLASSES)}  # CNV=0,DME=1,DRUSEN=2,NORMAL=3

# ImageNet 归一化参数
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# 弱语义文本模板
_WEAK_TEXT = {
    "NORMAL": [
        "normal retinal OCT scan with distinct layer boundaries",
        "healthy retina without fluid or lesions",
        "clear RPE and photoreceptor layers visible",
    ],
    "CNV": [
        "choroidal neovascularization with subretinal fluid",
        "irregular RPE elevation and subretinal hyperreflective material",
        "CNV membrane with fluid accumulation beneath retina",
    ],
    "DME": [
        "diabetic macular edema with intraretinal cysts",
        "retinal thickening and fluid in diabetic eye",
        "intraretinal hyperreflective foci consistent with DME",
    ],
    "DRUSEN": [
        "drusen deposits beneath RPE in macular region",
        "small hyperreflective mounds at RPE level",
        "early AMD changes with drusen accumulation",
    ],
}


# =====================================================================
# 核心工具：文件名前缀 → 真实类别
# =====================================================================
def parse_true_class(filepath: Path) -> Optional[str]:
    """
    从文件路径解析图像的真实类别。

    规则（优先级从高到低）：
      1. 目录名为 AMD → 必须从文件名前缀解析（如 CNV-1569-1.tif → CNV）
      2. 目录名直接是 CNV/DME/DRUSEN/NORMAL → 使用目录名
      3. 兜底：尝试文件名前缀解析

    文件名前缀匹配：文件名（去后缀、转大写）以 "类别-" 或 "类别_" 开头
    示例：
      CNV-1569-1.tif   → CNV
      DME-2301-5.tif   → DME
      DRUSEN-0003.tif  → DRUSEN
      NORMAL-001.tif   → NORMAL
    """
    parent = filepath.parent.name.upper()
    stem   = filepath.stem.upper()

    # 规则 2：目录名即类别（非 AMD 目录）
    if parent in TRUE_CLASSES:
        return parent

    # 规则 1 & 3：AMD 目录或其他未知目录 → 文件名前缀
    for cls in TRUE_CLASSES:
        if stem.startswith(cls + "-") or stem.startswith(cls + "_"):
            return cls
        if stem.startswith(cls):   # 兼容无分隔符
            return cls

    return None  # 无法解析


def _collect_images(base_dir: Path) -> List[Path]:
    """递归收集目录下所有图像文件"""
    paths = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        paths.extend(base_dir.rglob(ext))
    return paths


# =====================================================================
# 图像预处理 Pipeline
# =====================================================================
def build_transform(img_size: int = IMG_SIZE, augment: bool = False) -> transforms.Compose:
    steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if augment:
        # ── 强化增强：少标签场景相当于免费扩充训练集 ──────────────────
        # 注意：PIL 操作（Flip/Rotation/Affine/ColorJitter）须在 ToTensor 之前
        #       RandomErasing 作用于 Tensor，须在 ToTensor 之后
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.0,
                hue=0.0,
            ),
        ]
    steps += [
        transforms.ToTensor(),                                   # [0,1]
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),          # 1ch→3ch
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
    if augment:
        # RandomErasing 必须在 ToTensor 之后作用于 Tensor
        steps += [
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
    return transforms.Compose(steps)


# =====================================================================
# 1. 自监督预训练数据集（无标签，仅 NORMAL）
# =====================================================================
class PretrainDataset(Dataset):
    """
    自监督重建预训练数据集。

    【训练集 split="train"】
      - 仅加载 train/NORMAL/ 下的全部图像
      - 完全无标签，模型无法看到任何异常图像
      - 这是自监督学习的核心约束

    【验证集 split="val"】
      - 加载 val/ 下所有子目录（NORMAL + 异常）
      - 用于监控重建误差分布，验证正常/异常分离程度
      - 注意：此处仍无标签，误差图仅作可视化用

    返回: img_tensor [3, H, W]（ImageNet 归一化后）
    """

    def __init__(self,
                 oct_root: Path = OCT_IMAGE_ROOT,
                 split: str = "train",
                 img_size: int = IMG_SIZE):
        self.split    = split
        self.img_size = img_size
        base_dir      = Path(oct_root) / split

        if not base_dir.exists():
            raise FileNotFoundError(f"[PretrainDataset] 不存在: {base_dir}")

        if split == "train":
            # ✅ 只加载正常图像，严格无监督
            normal_dir = base_dir / "NORMAL"
            if not normal_dir.exists():
                normal_dir = base_dir / "normal"
            if not normal_dir.exists():
                raise FileNotFoundError(
                    f"[PretrainDataset] 找不到 NORMAL 目录: {base_dir}/NORMAL")
            self.img_paths = _collect_images(normal_dir)
            print(f"[PretrainDataset] train/NORMAL: {len(self.img_paths)} 张"
                  f" ← 无标签自监督，仅正常图像")
        else:
            self.img_paths = _collect_images(base_dir)
            print(f"[PretrainDataset] val (全部): {len(self.img_paths)} 张"
                  f" ← 含异常，用于验证误差分布")

        if not self.img_paths:
            raise FileNotFoundError(f"[PretrainDataset] {base_dir} 无图像")

        self.transform = build_transform(img_size, augment=False)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            return self.transform(Image.open(self.img_paths[idx]).convert("L"))
        except Exception as e:
            print(f"⚠️  加载失败 {self.img_paths[idx].name}: {e}")
            return torch.zeros(3, self.img_size, self.img_size)


# =====================================================================
# 2. 带标签分类数据集（含 AMD 目录文件名解析）
# =====================================================================
class LabeledOCTDataset(Dataset):
    """
    带标签 OCT 分类数据集，用于线性探测 / 有监督微调。

    ★ AMD 目录解析逻辑：
      AMD/ 目录中每个文件的真实类别由文件名前缀决定：
        CNV-1569-1.tif  → label 0 (CNV)
        DME-2301-5.tif  → label 1 (DME)
        DRUSEN-003.tif  → label 2 (DRUSEN)

    ★ 其他目录（CNV/DME/DRUSEN/NORMAL）直接用目录名作为标签。

    ★ 最终 4 类标签映射：
      CNV=0  DME=1  DRUSEN=2  NORMAL=3
    """

    CLASSES      = TRUE_CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX

    def __init__(self,
                 oct_root: Path = OCT_IMAGE_ROOT,
                 split: str = "val",
                 img_size: int = IMG_SIZE,
                 augment: bool = False):
        self.img_size  = img_size
        base_dir       = Path(oct_root) / split

        self.img_paths: List[Path] = []
        self.labels:    List[int]  = []
        parse_fail = 0

        for sub_dir in sorted(base_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            for p in _collect_images(sub_dir):
                cls = parse_true_class(p)
                if cls is None:
                    parse_fail += 1
                    continue
                self.img_paths.append(p)
                self.labels.append(CLASS_TO_IDX[cls])

        # ── 打印统计 ────────────────────────────────────────────────
        label_cnt = Counter(self.labels)
        print(f"\n[LabeledOCTDataset] {split}  总计: {len(self.img_paths)} 张"
              + (f"  (解析失败跳过: {parse_fail})" if parse_fail else ""))
        for cls in TRUE_CLASSES:
            idx = CLASS_TO_IDX[cls]
            print(f"  {cls:<8} {label_cnt.get(idx, 0):>6} 张  (label={idx})")

        if not self.img_paths:
            raise RuntimeError(f"[LabeledOCTDataset] {base_dir} 下未找到可用图像")

        self.transform = build_transform(img_size, augment=augment)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            t = self.transform(Image.open(self.img_paths[idx]).convert("L"))
            return t, self.labels[idx]
        except Exception as e:
            print(f"⚠️  加载失败 {self.img_paths[idx].name}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), self.labels[idx]


# =====================================================================
# 3. 多模态数据集（图像 + 弱语义文本，可选）
# =====================================================================
class MultimodalOCTDataset(Dataset):
    """
    多模态预训练数据集（无标签图像 + 弱语义/真实报告文本）。

    训练时同样仅使用 NORMAL 图像（split="train"），
    val 时使用全部图像以评估多模态表征。

    依赖：pip install transformers
    """

    def __init__(self,
                 oct_root: Path = OCT_IMAGE_ROOT,
                 split: str = "train",
                 img_size: int = IMG_SIZE,
                 text_csv_path: Optional[Path] = None,
                 tokenizer_name: str = TEXT_MODEL_NAME,
                 max_text_len: int = MAX_TEXT_LEN):
        self.img_size     = img_size
        self.max_text_len = max_text_len

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        base_dir = Path(oct_root) / split
        if split == "train":
            self.img_paths = _collect_images(base_dir / "NORMAL")
        else:
            self.img_paths = _collect_images(base_dir)

        # 文本来源
        self.text_map: Dict[str, str] = {}
        if text_csv_path and Path(text_csv_path).exists():
            import csv
            with open(text_csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.text_map[row["filepath"]] = row["text"]
            print(f"[MultimodalOCT] 真实报告: {len(self.text_map)} 条")
        else:
            print("[MultimodalOCT] 使用弱语义文本模板（按类别随机选取）")

        self.transform = build_transform(img_size, augment=False)
        print(f"[MultimodalOCT] {split}: {len(self.img_paths)} 张")

    def _get_text(self, p: Path) -> str:
        if str(p) in self.text_map:
            return self.text_map[str(p)]
        cls = parse_true_class(p) or "NORMAL"
        return random.choice(_WEAK_TEXT.get(cls, ["retinal OCT scan"]))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.img_paths[idx]
        try:
            img = self.transform(Image.open(p).convert("L"))
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)

        tok = self.tokenizer(
            self._get_text(p),
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return img, tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)


# =====================================================================
# 自检入口：python datasets.py
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("数据集解析自检（验证 AMD 目录文件名解析是否正确）")
    print("=" * 60)

    # 测试文件名解析
    test_cases = [
        ("AMD", "CNV-1569-1.tif"),
        ("AMD", "DME-2301-5.tif"),
        ("AMD", "DRUSEN-0003-1.tif"),
        ("CNV",    "some_file.tif"),
        ("NORMAL", "NORMAL-001.tif"),
    ]
    print("\n── 文件名解析单元测试 ──")
    for parent, fname in test_cases:
        p   = Path(f"/fake/{parent}/{fname}")
        cls = parse_true_class(p)
        status = "✅" if cls else "❌"
        print(f"  {status}  {parent}/{fname:<25s} → {cls}")

    print("\n── LabeledOCTDataset (val) ──")
    try:
        ds = LabeledOCTDataset(split="val")
        img, lbl = ds[0]
        print(f"  样本形状: {img.shape}  标签: {lbl} ({TRUE_CLASSES[lbl]})")
    except Exception as e:
        print(f"  跳过（{e}）")

    print("\n── PretrainDataset (train/NORMAL) ──")
    try:
        pt = PretrainDataset(split="train")
        print(f"  预训练样本: {len(pt)} 张  形状: {pt[0].shape}")
    except Exception as e:
        print(f"  跳过（{e}）")
