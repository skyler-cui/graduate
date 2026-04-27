"""
config.py — 全局超参数与路径配置
所有模块统一从此文件导入，避免硬编码分散。
"""
from pathlib import Path
import torch

# =====================================================================
# 路径配置
# =====================================================================
PROJECT_ROOT    = Path(__file__).resolve().parent
OCT_IMAGE_ROOT  = Path("/root/aimv2/oct_images")   # Kaggle OCT2017 根目录
PHOTO_SAVE_DIR  = Path("/root/aimv2/oct_results")
CKPT_SAVE_DIR   = PHOTO_SAVE_DIR / "checkpoints"

PHOTO_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 数据配置
# =====================================================================
IMG_SIZE             = 224
PATCH_SIZE           = 14
NUM_PATCHES          = (IMG_SIZE // PATCH_SIZE) ** 2   # 256
TRAIN_SAMPLE_RATIO   = 1.0          # 使用全部训练数据
VISUALIZE_NUM        = 20
# AMD 不是独立类别，只是容器目录，真实类别由文件名前缀决定
# 最终分类任务为 4 类
DISEASE_CLASSES      = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES          = len(DISEASE_CLASSES)   # 4类

# =====================================================================
# 文本模态配置
# =====================================================================
# 若无真实报告文本，可使用下方弱语义模板（见 datasets.py）
USE_TEXT_MODALITY    = True          # 关闭则退化为纯视觉自监督
MAX_TEXT_LEN         = 64            # token 最大长度
TEXT_MODEL_NAME      = "emilyalsentzer/Bio_ClinicalBERT"  # HuggingFace BERT

# =====================================================================
# 模型结构
# =====================================================================
VISION_DIM           = 768           # AIMv2-Large 特征维度
TEXT_DIM             = 768           # ClinicalBERT 隐层维度
FUSION_DIM           = 768           # 跨模态融合后维度
DECODER_LAYERS       = 6
DECODER_HEADS        = 6
AIM_PRETRAINED_NAME  = "aimv2-large-patch14-224-lit"

# 预训练阶段：完全冻结编码器，保护 ImageNet 判别特征不被自监督破坏
UNFREEZE_LAST_N_LAYERS         = 0
# 有监督微调阶段：单独控制解冻层数，与预训练互不影响
FINETUNE_UNFREEZE_LAST_N_LAYERS = 4

# =====================================================================
# 训练超参数
# =====================================================================
BATCH_SIZE           = 48
PATCH_EXTRACT_BATCH  = 64
TRAIN_EPOCHS         = 10
LINEAR_PROBE_EPOCHS  = 10
FINETUNE_EPOCHS      = 10

PRETRAIN_LR          = 5e-6
PRETRAIN_MAX_LR      = 2e-5
LINEAR_PROBE_LR      = 1e-3
FINETUNE_LR          = 5e-5   # 原 1e-4，降低防止灾难性遗忘

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

# =====================================================================
# AMP Scaler
# 修复：torch.cuda.amp.GradScaler() 在 PyTorch 2.0+ 已废弃，
#       改用 torch.amp.GradScaler('cuda') 统一接口。
# =====================================================================
if DEVICE.type == "cuda":
    try:
        # PyTorch >= 2.0 推荐写法
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        # 兼容旧版 PyTorch（< 2.0）
        scaler = torch.cuda.amp.GradScaler()   # type: ignore[attr-defined]
else:
    scaler = None
