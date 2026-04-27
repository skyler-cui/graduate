"""
config_boe.py — BOE Publication Dataset 专用超参数与路径配置

[BOE-CHANGE] 本文件是 config.py 的替换版本，专用于 BOE 三分类数据集。
             使用时将 config.py 替换为本文件（或修改 import），
             原 config.py 保持不变，以便切换回 Kaggle OCT2017 数据集。

修改点汇总（相对于 config.py）：
  [BOE-CHANGE-1]  数据根目录改为 boe_images/Publication_Dataset
  [BOE-CHANGE-2]  DISEASE_CLASSES 改为 3 类：AMD / DME / NORMAL
  [BOE-CHANGE-3]  NUM_CLASSES 改为 3
  [BOE-CHANGE-4]  新增 BOE_TRAIN_VAL_RATIO = 0.83（对齐 Kaggle OCT2017 比例）
  [BOE-CHANGE-5]  新增 BOE_RANDOM_SEED = 42（确保每次划分一致）
  [BOE-CHANGE-6]  移除 USE_TEXT_MODALITY（BOE 数据集无文本配对，暂不使用）
"""
from pathlib import Path
import torch

# =====================================================================
# 路径配置
# =====================================================================
PROJECT_ROOT    = Path(__file__).resolve().parent

# [BOE-CHANGE-1] 数据根目录：指向 BOE Publication_Dataset
BOE_IMAGE_ROOT  = Path("/root/aimv2/ml-aim/boe_images/Publication_Dataset")
PHOTO_SAVE_DIR  = Path("/root/aimv2/boe_results")
CKPT_SAVE_DIR   = PHOTO_SAVE_DIR / "checkpoints"

PHOTO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 兼容性别名（供不区分数据集的模块使用）
OCT_IMAGE_ROOT  = BOE_IMAGE_ROOT

# =====================================================================
# 数据配置
# =====================================================================
IMG_SIZE             = 224
PATCH_SIZE           = 14
NUM_PATCHES          = (IMG_SIZE // PATCH_SIZE) ** 2   # 256
TRAIN_SAMPLE_RATIO   = 1.0
VISUALIZE_NUM        = 20

# [BOE-CHANGE-2] 三分类：AMD / DME / NORMAL（无 DRUSEN 类）
DISEASE_CLASSES      = ["AMD", "DME", "NORMAL"]

# [BOE-CHANGE-3] 类别数改为 3
NUM_CLASSES          = len(DISEASE_CLASSES)   # 3 类

# [BOE-CHANGE-4] 训练/验证划分比例，对齐 Kaggle OCT2017 的自然比例约 83%/17%
BOE_TRAIN_VAL_RATIO  = 0.83

# [BOE-CHANGE-5] 固定随机种子，保证每次运行划分结果完全一致
BOE_RANDOM_SEED      = 42

# =====================================================================
# 文本模态配置
# [BOE-CHANGE-6] BOE 数据集无配对文本报告，禁用文本模态
# =====================================================================
USE_TEXT_MODALITY    = False
MAX_TEXT_LEN         = 64
TEXT_MODEL_NAME      = "emilyalsentzer/Bio_ClinicalBERT"

# =====================================================================
# 模型结构（与原 config.py 完全一致，无需修改）
# =====================================================================
VISION_DIM           = 768
TEXT_DIM             = 768
FUSION_DIM           = 768
DECODER_LAYERS       = 6
DECODER_HEADS        = 6
AIM_PRETRAINED_NAME  = "aimv2-large-patch14-224-lit"

UNFREEZE_LAST_N_LAYERS          = 0
FINETUNE_UNFREEZE_LAST_N_LAYERS = 4

# =====================================================================
# 训练超参数（与原 config.py 完全一致）
# =====================================================================
BATCH_SIZE           = 48
PATCH_EXTRACT_BATCH  = 64
TRAIN_EPOCHS         = 10
LINEAR_PROBE_EPOCHS  = 10
FINETUNE_EPOCHS      = 10

PRETRAIN_LR          = 5e-6
PRETRAIN_MAX_LR      = 2e-5
LINEAR_PROBE_LR      = 1e-3
FINETUNE_LR          = 5e-5

WEIGHT_DECAY         = 0.001
GRAD_CLIP            = 1.0

# =====================================================================
# 异常检测配置
# =====================================================================
ABNORMAL_THRESHOLD_PERCENTILE = 99
ROI_THRESHOLD_LOW             = 0.05
ROI_THRESHOLD_HIGH            = 0.95
ROI_MORPHOLOGY_KERNEL         = 3

# =====================================================================
# 设备
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    torch.backends.cudnn.benchmark         = True
    torch.backends.cudnn.deterministic     = False
    torch.set_float32_matmul_precision("high")

if DEVICE.type == "cuda":
    try:
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()   # type: ignore[attr-defined]
else:
    scaler = None
