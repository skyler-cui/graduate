"""
datasets_boe.py — BOE Publication Dataset 数据集定义

[BOE-CHANGE] 本文件是 datasets.py 的替换版本，专用于 BOE 三分类数据集。
             使用时将 datasets.py 替换为本文件（或修改 import），
             原 datasets.py 保持不变，以便切换回 Kaggle OCT2017 数据集。

BOE 数据集结构：
┌─ Publication_Dataset/
│   ├─ AMD1/TIFFs/8bitTIFFs/*.tif    ┐
│   ├─ AMD2/TIFFs/8bitTIFFs/*.tif    │  → 类别 AMD（标签 0）
│   ├─ ...                           │
│   ├─ AMD15/TIFFs/8bitTIFFs/*.tif   ┘
│   ├─ DME1/TIFFs/8bitTIFFs/*.tif    ┐
│   ├─ ...                           │  → 类别 DME（标签 1）
│   ├─ DME15/TIFFs/8bitTIFFs/*.tif   ┘
│   ├─ NORMAL1/TIFFs/8bitTIFFs/*.tif ┐
│   ├─ ...                           │  → 类别 NORMAL（标签 2）
│   └─ NORMAL15/TIFFs/8bitTIFFs/*.tif┘

与 datasets.py 的核心差异（[BOE-CHANGE] 标注）：
  [BOE-CHANGE-1]  从 config_boe 导入，而非 config
  [BOE-CHANGE-2]  TRUE_CLASSES 改为 3 类：AMD / DME / NORMAL
  [BOE-CHANGE-3]  parse_true_class：目录名前缀匹配（AMDx→AMD, DMEx→DME, NORMALx→NORMAL）
  [BOE-CHANGE-4]  _collect_images：路径深度适配 AMDx/TIFFs/8bitTIFFs/ 三层结构
  [BOE-CHANGE-5]  无预设 train/val 目录 → 新增 _split_indices() 按比例随机切分
  [BOE-CHANGE-6]  PretrainDataset：从所有 NORMAL 子目录收集图像，train/val 从切分结果取
  [BOE-CHANGE-7]  LabeledOCTDataset：同样依赖切分结果，split 参数含义不变
  [BOE-CHANGE-8]  MultimodalOCTDataset：同步适配（保留接口，实际 USE_TEXT_MODALITY=False）
"""

import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# [BOE-CHANGE-1] 从 config_boe 导入，替换原来的 from config import ...
from config_boe import (
    BOE_IMAGE_ROOT, IMG_SIZE,
    MAX_TEXT_LEN, TEXT_MODEL_NAME,
    BOE_TRAIN_VAL_RATIO, BOE_RANDOM_SEED,
)

# 兼容性别名，供未区分数据集的模块（如 evaluate.py）使用
OCT_IMAGE_ROOT = BOE_IMAGE_ROOT

# ── [BOE-CHANGE-2] 全局类别定义：3 类，去掉 DRUSEN ────────────────────────────
TRUE_CLASSES = ["AMD", "DME", "NORMAL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRUE_CLASSES)}  # AMD=0, DME=1, NORMAL=2

# ImageNet 归一化参数（不变）
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# 弱语义文本模板（保留接口，BOE 实验中不使用）
_WEAK_TEXT = {
    "NORMAL": [
        "normal retinal OCT scan with distinct layer boundaries",
        "healthy retina without fluid or lesions",
    ],
    "AMD": [
        "age-related macular degeneration with drusen deposits",
        "irregular RPE changes consistent with AMD",
    ],
    "DME": [
        "diabetic macular edema with intraretinal cysts",
        "retinal thickening and fluid in diabetic eye",
    ],
}


# =====================================================================
# [BOE-CHANGE-3] 核心工具：目录名前缀 → 真实类别
# 原版通过文件名前缀（CNV-xxx.tif）解析，
# BOE 版通过 **子目录名前缀**（AMD1, DME3, NORMAL12）解析。
# 路径示例：.../AMD1/TIFFs/8bitTIFFs/01.tif
#   → 祖先目录依次为 8bitTIFFs / TIFFs / AMD1 / Publication_Dataset
#   → 取 parts 中第一个以 TRUE_CLASSES 前缀开头的部分
# =====================================================================
def parse_true_class(filepath: Path) -> Optional[str]:
    """
    从文件路径中找到第一个以 AMD / DME / NORMAL 开头的目录段，
    返回对应的真实类别。

    示例：
      .../AMD1/TIFFs/8bitTIFFs/01.tif  → AMD
      .../DME12/TIFFs/8bitTIFFs/05.tif → DME
      .../NORMAL3/TIFFs/8bitTIFFs/9.tif → NORMAL
    """
    for part in filepath.parts:
        upper = part.upper()
        for cls in TRUE_CLASSES:
            if upper.startswith(cls):
                return cls
    return None


# [BOE-CHANGE-4] 适配三层深路径：AMDx/TIFFs/8bitTIFFs/
def _collect_images(base_dir: Path) -> List[Path]:
    """递归收集目录下所有图像文件（支持任意深度子目录）"""
    paths = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        paths.extend(base_dir.rglob(ext))
    return sorted(paths)  # 排序确保跨平台一致性


# =====================================================================
# [BOE-CHANGE-5] 全局 train/val 切分工具
# BOE 数据集无预设 train/val 目录，需在代码中按比例切分。
# 切分策略：**分层切分**（每个类别内部独立按比例切分），
#           保证 train/val 中类别比例一致。
# 结果缓存在模块级 dict，避免同一进程内重复随机（Dataset 实例化多次时一致）。
# =====================================================================
_SPLIT_CACHE: Dict[str, Tuple[List[Path], List[int], List[Path], List[int]]] = {}

def _build_split(
    dataset_root: Path,
    train_ratio:  float = BOE_TRAIN_VAL_RATIO,
    seed:         int   = BOE_RANDOM_SEED,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """
    收集 dataset_root 下所有图像，按类别分层随机切分为 train / val。

    返回：
      (train_paths, train_labels, val_paths, val_labels)
    """
    cache_key = f"{dataset_root}|{train_ratio}|{seed}"
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

    rng = random.Random(seed)

    # 按类别收集
    class_paths: Dict[str, List[Path]] = {cls: [] for cls in TRUE_CLASSES}
    for sub in sorted(dataset_root.iterdir()):
        if not sub.is_dir():
            continue
        cls = parse_true_class(sub)
        if cls is None:
            continue
        imgs = _collect_images(sub)
        class_paths[cls].extend(imgs)

    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    for cls, paths in class_paths.items():
        if not paths:
            print(f"[BOE Split] ⚠️ 类别 {cls} 无图像，跳过")
            continue
        rng.shuffle(paths)
        n_train = max(1, int(len(paths) * train_ratio))
        label   = CLASS_TO_IDX[cls]

        train_paths.extend(paths[:n_train])
        train_labels.extend([label] * n_train)
        val_paths.extend(paths[n_train:])
        val_labels.extend([label] * (len(paths) - n_train))

        print(f"[BOE Split] {cls:<8}: 总 {len(paths):>5} → "
              f"train {n_train:>4}  val {len(paths)-n_train:>4}")

    _SPLIT_CACHE[cache_key] = (train_paths, train_labels, val_paths, val_labels)
    return train_paths, train_labels, val_paths, val_labels


# =====================================================================
# 图像预处理 Pipeline（与 datasets.py 完全一致，无需修改）
# =====================================================================
def build_transform(img_size: int = IMG_SIZE, augment: bool = False) -> transforms.Compose:
    steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if augment:
        # ── 强化增强：少标签场景相当于免费扩充训练集 ──────────────────
        # 注意：PIL 操作须在 ToTensor 之前，RandomErasing 须在 ToTensor 之后
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
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
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
# [BOE-CHANGE-6] 自监督预训练数据集（无标签，仅 NORMAL）
# 原版：从 train/NORMAL/ 目录直接读取
# BOE 版：从全局切分结果中筛选 NORMAL 标签的路径
# =====================================================================
class PretrainDataset(Dataset):
    """
    自监督重建预训练数据集（无标签）。

    split="train": 使用全局切分后的 NORMAL 训练图像
    split="val"  : 使用全局切分后的全部验证图像（含异常），监控重建误差分布

    [BOE-CHANGE] 原版直接读取 train/NORMAL/ 目录；
                 BOE 版从 _build_split() 结果中按 label 筛选。
    """

    def __init__(self,
                 oct_root: Path = BOE_IMAGE_ROOT,
                 split: str = "train",
                 img_size: int = IMG_SIZE):
        self.split    = split
        self.img_size = img_size
        root          = Path(oct_root)

        train_paths, train_labels, val_paths, val_labels = _build_split(root)

        if split == "train":
            # 仅 NORMAL（label=2）图像用于自监督预训练
            normal_label = CLASS_TO_IDX["NORMAL"]
            self.img_paths = [p for p, l in zip(train_paths, train_labels)
                              if l == normal_label]
            print(f"[PretrainDataset] train/NORMAL: {len(self.img_paths)} 张"
                  f" ← 无标签自监督，仅正常图像")
        else:
            # val：全部类别，用于验证重建误差分布
            self.img_paths = val_paths
            print(f"[PretrainDataset] val (全部): {len(self.img_paths)} 张"
                  f" ← 含异常，用于验证误差分布")

        if not self.img_paths:
            raise FileNotFoundError(
                f"[PretrainDataset] {split} 集无图像，请检查数据路径: {root}")

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
# [BOE-CHANGE-7] 带标签分类数据集
# 原版：按 train/val 子目录 + AMD 文件名前缀解析
# BOE 版：直接从 _build_split() 结果中取对应 split 的路径与标签
# =====================================================================
class LabeledOCTDataset(Dataset):
    """
    带标签 OCT 分类数据集，用于线性探测 / 有监督微调。

    [BOE-CHANGE] 三分类：AMD=0  DME=1  NORMAL=2
                 split 的含义不变（"train" / "val"），
                 但数据来源改为 _build_split() 的切分结果。
    """

    CLASSES      = TRUE_CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX

    def __init__(self,
                 oct_root: Path = BOE_IMAGE_ROOT,
                 split: str = "val",
                 img_size: int = IMG_SIZE,
                 augment: bool = False):
        self.img_size = img_size
        root = Path(oct_root)

        train_paths, train_labels, val_paths, val_labels = _build_split(root)

        if split == "train":
            self.img_paths = train_paths
            self.labels    = train_labels
        else:
            self.img_paths = val_paths
            self.labels    = val_labels

        label_cnt = Counter(self.labels)
        print(f"\n[LabeledOCTDataset] {split}  总计: {len(self.img_paths)} 张")
        for cls in TRUE_CLASSES:
            idx = CLASS_TO_IDX[cls]
            print(f"  {cls:<8} {label_cnt.get(idx, 0):>6} 张  (label={idx})")

        if not self.img_paths:
            raise RuntimeError(
                f"[LabeledOCTDataset] {split} 集为空，请检查数据路径: {root}")

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
# [BOE-CHANGE-8] 多模态数据集（保留接口，BOE 实验中不启用）
# =====================================================================
class MultimodalOCTDataset(Dataset):
    """
    多模态预训练数据集（图像 + 文本）。

    [BOE-CHANGE] BOE 数据集无配对文本，USE_TEXT_MODALITY=False，
                 此类保留接口以保持与原 datasets.py 的 API 兼容，
                 实际训练中不会被调用。
    """

    def __init__(self,
                 oct_root: Path = BOE_IMAGE_ROOT,
                 split: str = "train",
                 img_size: int = IMG_SIZE,
                 text_csv_path=None,
                 tokenizer_name: str = TEXT_MODEL_NAME,
                 max_text_len: int = MAX_TEXT_LEN):
        self.img_size     = img_size
        self.max_text_len = max_text_len

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        root = Path(oct_root)
        train_paths, train_labels, val_paths, val_labels = _build_split(root)

        if split == "train":
            normal_label   = CLASS_TO_IDX["NORMAL"]
            self.img_paths = [p for p, l in zip(train_paths, train_labels)
                              if l == normal_label]
        else:
            self.img_paths = val_paths

        self.text_map: Dict[str, str] = {}
        if text_csv_path and Path(text_csv_path).exists():
            import csv
            with open(text_csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.text_map[row["filepath"]] = row["text"]

        self.transform = build_transform(img_size, augment=False)
        print(f"[MultimodalOCT-BOE] {split}: {len(self.img_paths)} 张")

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
# 自检入口：python datasets_boe.py
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BOE 数据集解析自检")
    print("=" * 60)

    print("\n── parse_true_class 单元测试 ──")
    test_cases = [
        "/data/AMD1/TIFFs/8bitTIFFs/01.tif",
        "/data/AMD15/TIFFs/8bitTIFFs/10.tif",
        "/data/DME3/TIFFs/8bitTIFFs/05.tif",
        "/data/NORMAL12/TIFFs/8bitTIFFs/09.tif",
        "/data/unknown/TIFFs/8bitTIFFs/xx.tif",
    ]
    for p in test_cases:
        cls = parse_true_class(Path(p))
        status = "✅" if cls else "❌"
        print(f"  {status}  {Path(p).parts[-4]:<15} → {cls}")

    print("\n── _build_split 切分统计 ──")
    try:
        tp, tl, vp, vl = _build_split(BOE_IMAGE_ROOT)
        print(f"  train: {len(tp)} 张  val: {len(vp)} 张")
        tc = Counter(tl); vc = Counter(vl)
        for cls in TRUE_CLASSES:
            idx = CLASS_TO_IDX[cls]
            print(f"  {cls:<8}: train {tc.get(idx,0):>4}  val {vc.get(idx,0):>4}")
    except Exception as e:
        print(f"  跳过（{e}）")

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
