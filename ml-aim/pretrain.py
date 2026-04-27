"""
pretrain.py — 自监督预训练训练器

支持两种训练模式：
  - Stage 1: 纯视觉自回归预训练（AIMv2OCTAnomalyDetector）
  - Stage 2: 多模态自回归预训练（MultimodalAIMv2）

用法：
    from pretrain import Trainer
    trainer = Trainer(model, train_ds, val_ds, stage=1)
    train_losses, val_losses = trainer.run()

修复记录：
  [BUG-5] val_epoch 中外层已有 torch.no_grad()，内层再套
          autocast()/no_grad() 逻辑混乱。
          → 移除外层 no_grad，内层统一使用 autocast（GPU）
            或 contextlib.nullcontext（CPU），结构清晰。
"""
import contextlib
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    DEVICE, scaler, BATCH_SIZE, TRAIN_EPOCHS,
    TRAIN_SAMPLE_RATIO, GRAD_CLIP,
    PRETRAIN_LR, PRETRAIN_MAX_LR, WEIGHT_DECAY,
    CKPT_SAVE_DIR
)


class Trainer:
    """
    通用自监督预训练训练器。
    stage=1 → 纯视觉模式（OCTDataset 返回单张 tensor）
    stage=2 → 多模态模式（MultimodalOCTDataset 返回 (img, ids, mask)）
    """

    def __init__(self, model: nn.Module,
                 train_dataset,
                 val_dataset,
                 stage: int = 1,
                 epochs: int = TRAIN_EPOCHS,
                 batch_size: int = BATCH_SIZE):
        self.model  = model
        self.stage  = stage
        self.epochs = epochs

        # 随机采样训练集（控制训练时长）
        n_train   = int(TRAIN_SAMPLE_RATIO * len(train_dataset))
        indices   = random.sample(range(len(train_dataset)), n_train)
        sampled   = Subset(train_dataset, indices)
        print(f"[Trainer] Stage-{stage} 采样训练集: {n_train}/{len(train_dataset)} 张")

        num_workers = 8
        self.train_loader = DataLoader(
            sampled, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4, persistent_workers=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4, persistent_workers=True, drop_last=False
        )

        # ── 优化器：涵盖所有可训练参数 ──────────────────────────────
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"[Trainer] 可训练参数量: {sum(p.numel() for p in trainable):,}")

        self.optimizer = torch.optim.AdamW(
            trainable, lr=PRETRAIN_LR,
            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=PRETRAIN_MAX_LR,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1, anneal_strategy='cos'
        )

        self.train_losses: list = []
        self.val_losses:   list = []

    # ── 解包 batch ─────────────────────────────────────────────────────
    def _unpack(self, batch):
        if self.stage == 1:
            img = batch[0] if isinstance(batch, (list, tuple)) else batch
            return img.to(DEVICE, non_blocking=True), None, None
        else:
            img, ids, attn = batch
            return (img.to(DEVICE, non_blocking=True),
                    ids.to(DEVICE, non_blocking=True),
                    attn.to(DEVICE, non_blocking=True))

    # ── 计算损失 ────────────────────────────────────────────────────────
    def _forward_loss(self, img, ids, attn):
        if self.stage == 1:
            return self.model.compute_loss(img)
        else:
            return self.model.compute_loss(img, ids, attn)

    # ── 单 epoch 训练 ───────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, count = 0.0, 0

        pbar = tqdm(self.train_loader,
                    desc=f"[Stage-{self.stage}] Epoch {epoch}/{self.epochs} Train")
        for batch in pbar:
            img, ids, attn = self._unpack(batch)
            self.optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = self._forward_loss(img, ids, attn)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), GRAD_CLIP)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss = self._forward_loss(img, ids, attn)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

            self.scheduler.step()
            total += loss.item(); count += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}",
                             avg=f"{total/count:.6f}")
        pbar.close()
        return total / count

    # ── 单 epoch 验证 ───────────────────────────────────────────────────
    def val_epoch(self, epoch: int) -> float:
        """
        [FIX-5] 原版在外层套 torch.no_grad()，内层再条件性套
                autocast() 或 no_grad()，逻辑混乱且有双重 no_grad 问题。
                修复：
                  - 统一在最外层 torch.no_grad() 阻断梯度。
                  - AMP autocast 单独控制混合精度，与 no_grad 平行嵌套。
                  - CPU 时 autocast_ctx 为 nullcontext，无副作用。
        """
        self.model.eval()
        total, count = 0.0, 0

        # AMP 上下文：GPU 时启用 autocast，CPU 时为空操作
        autocast_ctx = (torch.cuda.amp.autocast()
                        if scaler else contextlib.nullcontext())

        with torch.no_grad():
            pbar = tqdm(self.val_loader,
                        desc=f"[Stage-{self.stage}] Epoch {epoch}/{self.epochs} Val")
            for batch in pbar:
                img, ids, attn = self._unpack(batch)
                with autocast_ctx:
                    loss = self._forward_loss(img, ids, attn)
                total += loss.item(); count += 1
                pbar.set_postfix(val_loss=f"{total/count:.6f}")
            pbar.close()
        return total / count

    # ── 完整预训练流程 ──────────────────────────────────────────────────
    def run(self):
        print(f"\n🚀 开始 Stage-{self.stage} 预训练（{self.epochs} epochs）")

        for ep in range(1, self.epochs + 1):
            tl = self.train_epoch(ep)
            vl = self.val_epoch(ep)
            self.train_losses.append(tl)
            self.val_losses.append(vl)
            print(f"📊 Epoch {ep}: train={tl:.6f}  val={vl:.6f}\n")

        # 保存检查点
        ckpt_path = CKPT_SAVE_DIR / f"stage{self.stage}_pretrain.pth"
        torch.save({
            "stage":        self.stage,
            "epochs":       self.epochs,
            "model_state":  self.model.state_dict(),
            "optim_state":  self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
        }, ckpt_path)
        print(f"✅ Stage-{self.stage} 检查点保存: {ckpt_path}")

        return self.train_losses, self.val_losses
