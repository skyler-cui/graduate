"""
models.py — 模型定义

包含五个核心组件：
  1. AutoregressiveDecoder   — 基于 AIMv2 原生组件的自回归解码器
  2. AIMv2OCTAnomalyDetector — 纯视觉自监督预训练模型（Stage-1）
  3. MultimodalAIMv2         — 多模态（图像+文本）自回归预训练模型（Stage-2）
  4. LinearProbeClassifier   — 线性探测分类器（冻结编码器）
  5. FineTuneClassifier      — 有监督微调分类器（可解冻部分编码器）

依赖说明：
  - torch / torchvision ：必需
  - aim（aimv2）         ：必需，Stage-1/2 均依赖
  - transformers         ：可选，仅 MultimodalAIMv2（Stage-2）使用
                           未安装时 Stage-1 纯视觉功能完全正常

修复记录：
  [BUG-1] extract_patch_features 将 224×224 图像手动拆分为 14×14 小块后
          送入 encode_image，尺寸不匹配且特征无意义。
          → 改为传入完整图像，通过 trunk 获取 patch token 序列。
  [BUG-2] extract_patch_features 无条件使用 torch.no_grad()，
          导致训练时编码器解冻层梯度被截断，永远无法被优化。
          → 训练模式下使用 contextlib.nullcontext()，允许梯度回传。
  [BUG-3] 编码器总层数硬编码为 12，与实际模型可能不符。
          → 改为动态枚举 named_parameters，通过名称末尾数字推断总层数。
"""

# ── 标准库 ──────────────────────────────────────────────────────────────────
import re
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DEVICE, VISION_DIM, TEXT_DIM, FUSION_DIM,
    DECODER_LAYERS, DECODER_HEADS,
    PATCH_SIZE, NUM_PATCHES,
    AIM_PRETRAINED_NAME,
    UNFREEZE_LAST_N_LAYERS, FINETUNE_UNFREEZE_LAST_N_LAYERS,
    NUM_CLASSES,
)

# ── 可选依赖：aimv2 组件 ─────────────────────────────────────────────────────
try:
    from aim.v2.torch.layers import Attention, SwiGLUFFN, RMSNorm
    from aim.v2.utils import load_pretrained
    _AIM_OK = True
except ImportError:
    Attention = SwiGLUFFN = RMSNorm = load_pretrained = None  # type: ignore[assignment]
    _AIM_OK = False

# ── 可选依赖：transformers（仅 MultimodalAIMv2 使用）────────────────────────
try:
    from transformers import AutoModel
    _TRANSFORMERS_OK = True
except ImportError:
    AutoModel = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_OK = False


# =====================================================================
# 辅助函数
# =====================================================================
def create_causal_mask(x: torch.Tensor) -> torch.Tensor:
    """
    生成因果（下三角）掩膜 → [B, 1, L, L]
    确保自回归解码器中位置 t 只能看到 ≤t 的 token。
    """
    B, L, _ = x.shape
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    return mask.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)


def _infer_total_encoder_layers(named_params) -> tuple:
    """
    动态推断编码器总层数及层名前缀，支持多种 AIMv2 命名格式：

      格式 A（LiT / 标准 trunk）:
        preprocessor.trunk.blocks.{i}.*
        trunk.blocks.{i}.*

      格式 B（老版 AIMv2）:
        encoder.layers.{i}.*
        encoder.blocks.{i}.*

    返回: (total_layers: int, layer_prefix: str)
      layer_prefix 形如 "trunk.blocks" 或 "encoder.layers"，
      用于后续 `if layer_prefix + f".{i}" in name` 的解冻判断。
    """
    # 候选模式：(正则, 对应的层名前缀模板)
    # 注意：优先匹配更具体的前缀，避免子串误匹配
    CANDIDATES = [
        (re.compile(r"image_encoder\.trunk\.blocks\.(\d+)"),  "image_encoder.trunk.blocks"),
        (re.compile(r"preprocessor\.trunk\.blocks\.(\d+)"),   "preprocessor.trunk.blocks"),
        (re.compile(r"(?<!\w)trunk\.blocks\.(\d+)"),          "trunk.blocks"),
        (re.compile(r"encoder\.blocks\.(\d+)"),               "encoder.blocks"),
        (re.compile(r"encoder\.layers\.(\d+)"),               "encoder.layers"),
    ]

    # 把 generator 转成 list，因为要多次遍历
    params = list(named_params)

    for pattern, prefix in CANDIDATES:
        max_idx = -1
        for name, _ in params:
            m = pattern.search(name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx >= 0:
            total = max_idx + 1
            print(f"   动态识别编码器层名格式: '{prefix}.{{i}}'，共 {total} 层")
            return total, prefix

    # ── 所有模式均未命中，打印前 20 个参数名辅助诊断 ──────────────────
    print("⚠️  无法自动识别编码器层名格式，前 20 个参数名如下（请手动确认）：")
    for name, _ in params[:20]:
        print(f"     {name}")
    print("   使用默认值: 层数=24，前缀='image_encoder.trunk.blocks'")
    return 24, "image_encoder.trunk.blocks"


# =====================================================================
# 1. 自回归解码器（纯视觉 & 多模态共用）
# =====================================================================
class AutoregressiveDecoder(nn.Module):
    """
    6 层 Transformer 解码器，直接复用 AIMv2 原生组件
    （Attention / SwiGLUFFN / RMSNorm），训练稳定无需额外实现。
    """

    def __init__(self,
                 dim: int = VISION_DIM,
                 num_heads: int = DECODER_HEADS,
                 num_layers: int = DECODER_LAYERS):
        super().__init__()

        if not _AIM_OK:
            raise ImportError(
                "aimv2 未安装，请先执行: pip install aim  "
                "或将 aimv2 目录添加到 PYTHONPATH"
            )

        _Attention = Attention
        _SwiGLU    = SwiGLUFFN
        _RMSNorm   = RMSNorm

        class _Block(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.norm1 = _RMSNorm(dim)
                self_inner.attn  = _Attention(dim=dim, num_heads=num_heads, use_bias=True)
                self_inner.norm2 = _RMSNorm(dim)
                self_inner.ffn   = _SwiGLU(in_features=dim, hidden_features=dim * 4)

            def forward(self_inner, x: torch.Tensor,
                        mask: torch.Tensor = None) -> torch.Tensor:
                x = x + self_inner.attn(self_inner.norm1(x), mask=mask)
                x = x + self_inner.ffn(self_inner.norm2(x))
                return x

        self.layers = nn.ModuleList([_Block() for _ in range(num_layers)])
        self.norm   = RMSNorm(dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


# =====================================================================
# 2. 纯视觉自监督预训练模型（Stage-1）
# =====================================================================
class AIMv2OCTAnomalyDetector(nn.Module):
    """
    以 AIMv2-Large LiT 为骨干的自回归特征重建模型。

    训练目标：给定前 t 个 patch 的特征，预测第 t+1 个 patch 的特征（MSE）。
    推断阶段：逐 patch 重建误差作为异常分数，ROI 掩膜后统计。

    修复说明（相对原版）：
      - extract_patch_features：传入完整 224×224 图像，经 AIMv2 trunk 前向
        得到 patch token 序列，不再错误地拆分后送入 encode_image。
      - 训练时（self.training=True）不使用 no_grad，梯度可正常流回解冻层。
      - 编码器层数动态推断，不再硬编码为 12。
    """

    def __init__(self):
        super().__init__()

        if not _AIM_OK:
            raise ImportError("aimv2 未安装，请先安装或配置 PYTHONPATH")

        print(f"📥 加载预训练 AIMv2 LiT ({AIM_PRETRAINED_NAME})")
        self.lit_model = load_pretrained(AIM_PRETRAINED_NAME, backend="torch")
        self.lit_model.to(DEVICE).eval()
        print("✅ AIMv2 LiT 加载成功")
        # ── [FIX-3] 动态推断总层数和层名前缀，不再硬编码 ──────────────
        total_layers, layer_prefix = _infer_total_encoder_layers(
            self.lit_model.named_parameters())

        # 冻结全部参数，再按层号解冻最后 N 层
        for name, param in self.lit_model.named_parameters():
            param.requires_grad = False
            for i in range(total_layers - UNFREEZE_LAST_N_LAYERS, total_layers):
                if f"{layer_prefix}.{i}." in name:
                    param.requires_grad = True

        unfrozen = sum(p.numel() for p in self.lit_model.parameters()
                       if p.requires_grad)
        print(f"✅ 解冻最后 {UNFREEZE_LAST_N_LAYERS} 层（共 {total_layers} 层，"
              f"前缀='{layer_prefix}'），可训练参数: {unfrozen:,}")

        # ── 动态探测真实特征维度（避免与 config.VISION_DIM 不一致） ─────
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224, device=DEVICE)
            _enc   = getattr(self.lit_model, "image_encoder", None)
            if _enc is not None:
                _pre = getattr(_enc, "preprocessor", None)
                _trk = getattr(_enc, "trunk", None)
                if _pre is not None and _trk is not None:
                    _t = _pre(_dummy)
                    if isinstance(_t, tuple): _t = _t[0]
                    _o = _trk(_t)
                    if isinstance(_o, tuple): _o = _o[0]
                else:
                    _o = _enc(_dummy)
                    if isinstance(_o, tuple): _o = _o[0]
            else:
                _o = self.lit_model.trunk(_dummy)
                if isinstance(_o, tuple): _o = _o[0]
            actual_dim = int(_o.shape[-1])

        if actual_dim != VISION_DIM:
            print(f"⚠️  实际特征维度 {actual_dim} != config.VISION_DIM {VISION_DIM}，"
                  f"以实际值 {actual_dim} 为准。")
        self.dim = actual_dim

        # 自动计算能整除 actual_dim 的 head 数：
        # 从 DECODER_HEADS 开始，双向（先向上再向下）找最近的因数，
        # 优先选更大的 head 数以保证模型容量。
        def _find_heads(dim: int, preferred: int) -> int:
            for delta in range(0, preferred + 1):
                for h in (preferred + delta, preferred - delta):
                    if h > 0 and dim % h == 0:
                        return h
            return 1

        num_heads = _find_heads(self.dim, DECODER_HEADS)
        if num_heads != DECODER_HEADS:
            print(f"⚠️  DECODER_HEADS={DECODER_HEADS} 无法整除 dim={self.dim}，"
                  f"自动调整为 num_heads={num_heads}。"
                  f"（建议在 config.py 中将 DECODER_HEADS 设为 8 或 16）")

        self.start_token = nn.Parameter(
            torch.randn(1, 1, self.dim, device=DEVICE))
        self.decoder     = AutoregressiveDecoder(
            dim=self.dim, num_heads=num_heads).to(DEVICE)
        self.recon_head  = nn.Sequential(
            RMSNorm(self.dim),
            nn.Linear(self.dim, self.dim)
        ).to(DEVICE)

    # ── 特征提取 ──────────────────────────────────────────────────────
    def extract_patch_features(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        提取每张图像的 patch token 序列（AIMv2 trunk 前向）。

        [FIX-1] 原版将图像手动拆为 256 个 14×14 小块再送 encode_image，
                尺寸不匹配导致特征无意义或直接崩溃。
                现改为将完整 224×224 图像送入 trunk，直接获得 patch token 序列。

        [FIX-2] 原版无条件 torch.no_grad()，编码器解冻层梯度被截断。
                现在训练模式（self.training=True）下使用 nullcontext()，
                梯度可正常回传；推断模式下仍阻断梯度节省显存。

        Args:  img_batch [B, 3, 224, 224]
        Returns: [B, NUM_PATCHES, VISION_DIM]  即 [B, 256, 768]
        """
        # 训练时允许梯度流回解冻层；推断时阻断以节省显存
        grad_ctx = (contextlib.nullcontext() if self.training
                    else torch.no_grad())

        with grad_ctx:
            with torch.cuda.amp.autocast():
                imgs = img_batch.to(DEVICE, non_blocking=True)

                # ── 按实际模型结构逐级调用 ────────────────────────────
                # 优先尝试 image_encoder（LiT 格式：含 preprocessor + trunk）
                # 回退尝试裸 trunk（其他 AIMv2 变体）
                enc = getattr(self.lit_model, "image_encoder", None)
                if enc is not None:
                    pre  = getattr(enc, "preprocessor", None)
                    trk  = getattr(enc, "trunk", None)
                    if pre is not None and trk is not None:
                        # LiT 标准路径：preprocessor → trunk
                        tokens    = pre(imgs)
                        # 部分版本返回 tuple，取第一个元素
                        if isinstance(tokens, tuple):
                            tokens = tokens[0]
                        trunk_out = trk(tokens)
                        if isinstance(trunk_out, tuple):
                            trunk_out = trunk_out[0]
                    else:
                        trunk_out = enc(imgs)
                        if isinstance(trunk_out, tuple):
                            trunk_out = trunk_out[0]
                else:
                    trunk_out = self.lit_model.trunk(imgs)
                    if isinstance(trunk_out, tuple):
                        trunk_out = trunk_out[0]

        # 若 trunk 输出含 CLS token（位置 0），截去之
        if trunk_out.shape[1] == NUM_PATCHES + 1:
            trunk_out = trunk_out[:, 1:, :]   # [B, NUM_PATCHES, D]

        # 维度校验
        assert trunk_out.shape[1] == NUM_PATCHES, (
            f"[extract_patch_features] 期望 {NUM_PATCHES} 个 patch token，"
            f"实际得到 {trunk_out.shape[1]}。"
            f"请检查 AIMv2 trunk 的输出格式。"
        )
        return trunk_out   # [B, 256, 768]

    # ── 并行自回归预测（teacher-forcing） ─────────────────────────────
    def autoregressive_predict(self, img_batch: torch.Tensor) -> torch.Tensor:
        """与训练 loss 计算完全对齐的并行推断。Returns [B, 256, 768]"""
        B          = img_batch.shape[0]
        true_feats = self.extract_patch_features(img_batch)
        start      = self.start_token.to(true_feats.device).expand(B, -1, -1)
        input_seq  = torch.cat([start, true_feats[:, :-1]], dim=1)
        mask       = create_causal_mask(input_seq)
        with torch.cuda.amp.autocast():
            return self.recon_head(self.decoder(input_seq, mask=mask))

    # ── 训练损失（MSE） ───────────────────────────────────────────────
    def compute_loss(self, img_batch: torch.Tensor) -> torch.Tensor:
        true_feats = self.extract_patch_features(img_batch)
        B          = img_batch.shape[0]
        start      = self.start_token.to(true_feats.device).expand(B, -1, -1)
        input_seq  = torch.cat([start, true_feats[:, :-1]], dim=1)
        mask       = create_causal_mask(input_seq)
        with torch.cuda.amp.autocast():
            pred = self.recon_head(self.decoder(input_seq, mask=mask))
            return F.mse_loss(pred, true_feats)

    # ── 旧接口兼容别名 ────────────────────────────────────────────────
    compute_reconstruction_loss     = compute_loss
    autoregressive_predict_features = autoregressive_predict
    extract_patch_level_features    = extract_patch_features


# =====================================================================
# 3. 多模态自回归预训练模型（Stage-2）
# =====================================================================
class MultimodalAIMv2(nn.Module):
    """
    多模态自回归预训练模型（图像 patch + 文本 token 联合序列）。

    序列构造：
        [START] [text_tok_1 .. text_tok_N] [img_patch_0 .. img_patch_254]
    训练目标：视觉重建 MSE（文本 token 作为解剖语义前缀上下文）

    BUG 继承修复：
      - 依赖 AIMv2OCTAnomalyDetector.extract_patch_features，
        该方法已修复尺寸问题和梯度截断问题，此处无需额外改动。
    """

    def __init__(self, vision_model: "AIMv2OCTAnomalyDetector"):
        super().__init__()

        if not _TRANSFORMERS_OK or AutoModel is None:
            raise ImportError(
                "transformers 未安装，Stage-2 不可用。\n"
                "安装: pip install transformers\n"
                "或运行时添加 --skip-multimodal 跳过此阶段"
            )

        self.vision_model = vision_model
        # 以视觉模型探测到的真实维度为准
        self.dim          = vision_model.dim

        # ── 文本编码器（冻结） ────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.to(DEVICE)
        print("✅ ClinicalBERT 文本编码器加载成功（已冻结）")

        # ── 跨模态投影层（可训练） ────────────────────────────────────
        self.text_proj  = nn.Linear(TEXT_DIM, self.dim).to(DEVICE)
        self.img_proj   = nn.Identity().to(DEVICE)

        # 自动计算能整除 self.dim 的 head 数（与 AIMv2OCTAnomalyDetector 逻辑一致）
        def _find_heads(dim: int, preferred: int) -> int:
            for delta in range(0, preferred + 1):
                for h in (preferred + delta, preferred - delta):
                    if h > 0 and dim % h == 0:
                        return h
            return 1

        num_heads = _find_heads(self.dim, DECODER_HEADS)
        if num_heads != DECODER_HEADS:
            print(f"⚠️  MultimodalAIMv2: DECODER_HEADS={DECODER_HEADS} 无法整除 "
                  f"dim={self.dim}，自动调整为 num_heads={num_heads}。")

        # ── 共享解码器 ────────────────────────────────────────────────
        self.decoder    = AutoregressiveDecoder(
            dim=self.dim, num_heads=num_heads).to(DEVICE)
        self.recon_head = nn.Sequential(
            RMSNorm(self.dim),
            nn.Linear(self.dim, self.dim)
        ).to(DEVICE)
        self.start_token = nn.Parameter(
            torch.randn(1, 1, self.dim, device=DEVICE))

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """文本 token 序列编码 → [B, L_text, FUSION_DIM]"""
        with torch.no_grad():
            text_feats = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state                    # [B, L, TEXT_DIM]
        return self.text_proj(text_feats)          # [B, L, FUSION_DIM]

    def compute_loss(self,
                     img_batch:      torch.Tensor,
                     input_ids:      torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """多模态自回归损失（视觉侧 MSE；文本侧提供解剖语义上下文）"""
        B          = img_batch.shape[0]
        img_feats  = self.img_proj(
            self.vision_model.extract_patch_features(img_batch))   # [B, 256, D]
        txt_feats  = self.encode_text(input_ids, attention_mask)   # [B, L,   D]

        start      = self.start_token.to(img_feats.device).expand(B, -1, -1)
        input_seq  = torch.cat([start, txt_feats, img_feats[:, :-1]], dim=1)
        mask       = create_causal_mask(input_seq)

        with torch.cuda.amp.autocast():
            dec_out  = self.decoder(input_seq, mask=mask)
            L_txt    = txt_feats.shape[1]
            img_pred = self.recon_head(dec_out[:, 1 + L_txt:, :])  # [B, 255, D]
            return F.mse_loss(img_pred, img_feats[:, 1:, :])


# =====================================================================
# 4. 线性探测分类器
# =====================================================================
class LinearProbeClassifier(nn.Module):
    """
    冻结视觉编码器全部参数，只训练单层线性分类头。
    准确率 / AUC 直接反映预训练表征的判别能力。
    """

    def __init__(self,
                 pretrained_model: "AIMv2OCTAnomalyDetector",
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder    = pretrained_model
        actual_dim      = pretrained_model.dim
        self.classifier = nn.Linear(actual_dim, num_classes)

        for p in self.encoder.parameters():
            p.requires_grad = False
        print(f"✅ LinearProbeClassifier: {actual_dim} → {num_classes}（编码器已冻结）")

    def extract_global_feat(self, img_batch: torch.Tensor) -> torch.Tensor:
        """patch 特征均值池化 → 全局向量 [B, actual_dim]"""
        return self.encoder.extract_patch_features(img_batch).mean(dim=1)

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        # 线性探测时编码器始终冻结，显式 no_grad 节省显存
        with torch.no_grad():
            feat = self.extract_global_feat(img_batch)
        return self.classifier(feat)


# =====================================================================
# 5. 有监督微调分类器
# =====================================================================
class FineTuneClassifier(nn.Module):
    """
    有监督微调分类器（预训练 → 微调范式）。
    解冻编码器最后 unfreeze_layers 层 + MLP 分类头。

    相比线性探测：非线性头 + 微调编码器，性能上界更高。
    若标注数据少（<500 张），建议设 freeze_all_encoder=True。

    修复：编码器总层数改为动态推断，不再硬编码为 12。
    """

    def __init__(self,
                 pretrained_model: "AIMv2OCTAnomalyDetector",
                 num_classes: int = NUM_CLASSES,
                 unfreeze_layers: int = FINETUNE_UNFREEZE_LAST_N_LAYERS,
                 freeze_all_encoder: bool = False):
        super().__init__()
        self.encoder = pretrained_model
        actual_dim   = pretrained_model.dim

        if freeze_all_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("✅ FineTuneClassifier: 编码器完全冻结")
        else:
            total_layers, layer_prefix = _infer_total_encoder_layers(
                self.encoder.lit_model.named_parameters())

            for name, p in self.encoder.lit_model.named_parameters():
                p.requires_grad = False
                for i in range(total_layers - unfreeze_layers, total_layers):
                    if f"{layer_prefix}.{i}." in name:
                        p.requires_grad = True
            print(f"✅ FineTuneClassifier: 解冻最后 {unfreeze_layers} 层"
                  f"（共 {total_layers} 层，前缀='{layer_prefix}'）")

        # ── 增强分类头：三层MLP + 较强Dropout，少标签场景防过拟合 ──
        # Dropout 0.1→0.3，增加中间层维度梯级，提升非线性表达
        mid_dim = actual_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(actual_dim),
            nn.Linear(actual_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(0.3),          # 加强正则，少标签场景关键
            nn.Linear(mid_dim, mid_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid_dim // 2, num_classes),
        )
        print(f"   分类头: {actual_dim} → {mid_dim} → {mid_dim//2} → {num_classes}"
              f"  (Dropout=0.3/0.2)")

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.extract_patch_features(img_batch).mean(dim=1)
        return self.head(feat)
