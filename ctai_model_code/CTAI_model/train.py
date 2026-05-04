# -*- coding: utf-8 -*-
"""
CTAI 完整训练脚本
- AdamW + CosineAnnealingWarmRestarts
- 梯度裁剪 + EMA
- Deep Supervision 多尺度损失
- 定期滑窗评估 + 可视化保存
- 早停机制
"""

import os
import sys
import random
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目内部导入
from config import TrainConfig, get_config_from_args
from data.dataset import CTTumorDataset, CTFullImageDataset, scan_data_directory, split_by_patient, print_data_report
from core.losses import get_loss_fn, CombinedLoss
from core.ema import EMA
from inference import sliding_window_inference, postprocess, _dice_score, _precision_recall


# ============================================================
# 随机种子
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 模型初始化
# ============================================================

def build_model(config) -> nn.Module:
    """根据配置构建模型"""
    if config.model == "attention_unet":
        from net.attention_unet import AttentionUnet
        model = AttentionUnet(
            in_ch=config.in_channels,
            out_ch=config.out_channels,
            deep_supervision=config.deep_supervision,
            dropout_rate=config.dropout_rate
        )
        print(f"[模型] Attention U-Net, DS={config.deep_supervision}, Dropout={config.dropout_rate}")
    elif config.model == "unet":
        from net.unet import Unet
        model = Unet(in_ch=config.in_channels, out_ch=config.out_channels)
        print(f"[模型] 标准 U-Net")
    else:
        raise ValueError(f"未知的模型类型: {config.model}")
    return model


def weights_init(m):
    """Kaiming 权重初始化（适配 ReLU 激活函数）"""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


# ============================================================
# 深监督损失计算
# ============================================================

def compute_ds_loss(outputs, targets, loss_fn, ds_weights=(0.6, 0.2, 0.1, 0.1)):
    """
    深监督损失: 对多尺度输出加权求和
    outputs: [final_pred, ds1, ds2, ds3]
    ds_weights: 各层权重
    """
    total_loss_info = None

    for i, (pred, weight) in enumerate(zip(outputs, ds_weights)):
        if isinstance(loss_fn, CombinedLoss):
            info = loss_fn(pred, targets)
            if total_loss_info is None:
                total_loss_info = {k: v * weight if k == 'total' else v for k, v in info.items()}
            else:
                total_loss_info['total'] = total_loss_info['total'] + info['total'] * weight
        else:
            loss = loss_fn(pred, targets)
            if total_loss_info is None:
                total_loss_info = {'total': loss * weight}
            else:
                total_loss_info['total'] = total_loss_info['total'] + loss * weight

    return total_loss_info


# ============================================================
# 评估函数
# ============================================================

def evaluate(model, eval_dataset, config, device, epoch, vis_count=3, fast=True):
    """
    滑窗推理评估
    fast=True: 只评估含肿瘤切片（最多 val_max_slices 张），大 stride 加速
    fast=False: 全量评估，正常 stride
    """
    model.eval()
    dice_scores = []
    prec_scores = []
    recall_scores = []
    tumor_dice_scores = []
    vis_saved = 0

    stride = config.eval_stride * 2 if fast else config.eval_stride  # 快速模式步长翻倍

    # 决定评估哪些切片
    if fast:
        # 快速模式：只评估含肿瘤切片，最多 val_max_slices 张
        tumor_indices = [i for i, s in enumerate(eval_dataset.samples) if s['has_tumor']]
        if len(tumor_indices) > config.val_max_slices:
            rng = np.random.RandomState(epoch)  # 每个 epoch 采样不同子集
            tumor_indices = rng.choice(tumor_indices, config.val_max_slices, replace=False).tolist()
        eval_indices = tumor_indices
    else:
        eval_indices = list(range(len(eval_dataset)))

    with torch.no_grad():
        for idx in eval_indices:
            image_tensor, mask_tensor, person_id, slice_id = eval_dataset[idx]
            gt_mask = mask_tensor.squeeze().numpy()
            has_tumor = gt_mask.sum() > 0

            prob_map = sliding_window_inference(
                model, image_tensor,
                config.eval_patch_size, stride, device
            )
            pred_binary = postprocess(prob_map, config.threshold, config.min_area, config.max_area)

            dice = _dice_score(pred_binary, gt_mask)
            precision, recall = _precision_recall(pred_binary, gt_mask)

            dice_scores.append(dice)
            prec_scores.append(precision)
            recall_scores.append(recall)
            if has_tumor:
                tumor_dice_scores.append(dice)

            if vis_saved < vis_count and has_tumor:
                _save_eval_vis(
                    image_tensor.squeeze().numpy(), gt_mask, prob_map, pred_binary,
                    epoch, person_id, slice_id, dice, config.vis_dir
                )
                vis_saved += 1

    avg_dice = np.mean(tumor_dice_scores) if tumor_dice_scores else 0.0
    avg_prec = np.mean(prec_scores)
    avg_recall = np.mean(recall_scores)

    mode_str = "Fast" if fast else "Full"
    print(f"  [{mode_str} Eval] Epoch {epoch+1}: "
          f"TumorDice={avg_dice:.4f} ({len(tumor_dice_scores)} slices), "
          f"Precision={avg_prec:.4f}, Recall={avg_recall:.4f}")

    return avg_dice, avg_prec, avg_recall


def _save_eval_vis(image_np, gt_mask, prob_map, pred_binary, epoch, pid, sid, dice, vis_dir):
    """保存评估可视化"""
    try:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('GT Mask')
        axes[1].axis('off')

        im = axes[2].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Pred Prob')
        axes[2].axis('off')

        axes[3].imshow(pred_binary, cmap='gray')
        axes[3].set_title(f'Binary (Dice={dice:.3f})')
        axes[3].axis('off')

        plt.suptitle(f'Epoch {epoch+1} | {pid} | {sid}')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'epoch_{epoch+1}_{pid}_{sid}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARNING] 保存可视化失败: {e}")


# ============================================================
# 训练曲线绘制
# ============================================================

def draw_curves(history, save_dir):
    """绘制训练损失和 Dice 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dice 曲线
    eval_epochs = history['eval_epoch']
    if eval_epochs:
        ax2.plot(eval_epochs, history['eval_dice'], 'r-o', label='Eval Dice', linewidth=2)
        ax2.plot(eval_epochs, history['eval_precision'], 'g--', label='Precision', alpha=0.7)
        ax2.plot(eval_epochs, history['eval_recall'], 'b--', label='Recall', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Evaluation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")


# ============================================================
# 主训练循环
# ============================================================

def train(config: TrainConfig):
    """完整训练流程"""
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    use_multi_gpu = n_gpus >= 2
    print(f"[设备] {device}  |  GPU 数量: {n_gpus}")
    if device.type == 'cuda':
        for i in range(n_gpus):
            print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()

    # ==================== 数据（患者级 train/val 划分）====================
    print("\n" + "=" * 60)
    print("扫描并划分数据...")
    all_samples = scan_data_directory(config.data_dir)
    if not all_samples:
        raise RuntimeError(f"在 {config.data_dir} 中未找到任何有效样本！")
    print_data_report(all_samples)

    train_samples, val_samples = split_by_patient(all_samples, config.val_ratio, config.seed)

    print("\n加载训练数据...")
    train_dataset = CTTumorDataset(config.data_dir, config, mode="train", samples=train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    print("\n加载评估数据...")
    eval_dataset = CTFullImageDataset(config.data_dir, config, samples=val_samples)
    print(f"评估数据: {len(eval_dataset)} 张全尺寸切片 (来自独立验证患者)")

    # helper: 去掉 DataParallel 的 "module." 前缀
    def _strip_dp(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        return {(k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()}

    def _strip_dp_k(key):
        return key[7:] if key.startswith('module.') else key

    # ==================== 模型 ====================
    model = build_model(config)
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"[模型] DataParallel 已启用，{n_gpus} 块 GPU 并行")
    model = model.to(device)
    model.apply(weights_init)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[参数] 总量: {total_params:,}, 可训练: {trainable_params:,}")

    # ==================== 损失 / 优化器 / 调度器 ====================
    loss_fn = get_loss_fn(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.scheduler == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.T_0, eta_min=config.eta_min
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.eta_min
        )
    elif config.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
    else:
        scheduler = None

    # ==================== EMA ====================
    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None

    # ==================== 训练记录 ====================
    history = {
        'epoch': [], 'train_loss': [], 'lr': [],
        'eval_epoch': [], 'eval_dice': [], 'eval_precision': [], 'eval_recall': []
    }
    best_dice = -1.0
    no_improve_count = 0
    start_epoch = 0

    # ==================== 断点恢复 ====================
    if config.resume and os.path.isfile(config.resume):
        print(f"\n正在从 checkpoint 恢复: {config.resume}")
        ckpt = torch.load(config.resume, map_location=device)
        model_sd = _strip_dp(ckpt['model_state_dict'])
        _raw = model.module if use_multi_gpu else model
        _raw.load_state_dict(model_sd)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'ema_state_dict' in ckpt and ema is not None:
            ema.shadow = _strip_dp({k: v.to(device) for k, v in ckpt['ema_state_dict'].items()})
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']  # checkpoint 保存的是已完成的 epoch 数
        if 'best_dice' in ckpt:
            best_dice = ckpt['best_dice']
        if 'no_improve_count' in ckpt:
            no_improve_count = ckpt['no_improve_count']
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"  ✓ 已恢复到 Epoch {start_epoch}, Best Dice={best_dice:.4f}")
        print(f"  ✓ 将从 Epoch {start_epoch + 1} 继续训练")
    elif config.resume:
        print(f"[WARNING] checkpoint 文件不存在: {config.resume}，将从头开始训练")

    # ==================== 训练循环 ====================
    print("\n" + "=" * 60)
    print(f"开始训练: Epoch {start_epoch + 1} → {config.epochs}, batch_size={config.batch_size}")
    print("=" * 60)

    for epoch in range(start_epoch, config.epochs):
        model.train()

        # 设置损失函数的 epoch（用于 warm-up）
        if isinstance(loss_fn, CombinedLoss):
            loss_fn.set_epoch(epoch)

        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)

        for batch_idx, (images, masks, pids, sids) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            masks = (masks > 0).float()

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            if isinstance(outputs, list):
                # Deep Supervision: 多尺度损失
                loss_info = compute_ds_loss(outputs, masks, loss_fn)
            else:
                if isinstance(loss_fn, CombinedLoss):
                    loss_info = loss_fn(outputs, masks)
                else:
                    loss_val = loss_fn(outputs, masks)
                    loss_info = {'total': loss_val}

            loss = loss_info['total']

            # NaN/Inf 保护：跳过坏 batch，防止模型权重被污染
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] Epoch {epoch+1} batch {batch_idx}: loss={loss.item()}, 跳过此 batch")
                optimizer.zero_grad()  # 清除可能的脏梯度
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            # EMA 更新
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        # Epoch 结束
        avg_loss = epoch_loss / max(len(train_loader), 1)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['lr'].append(current_lr)

        # 学习率调度
        if scheduler is not None:
            if config.scheduler == "reduce_on_plateau":
                pass  # 在评估后调用
            else:
                scheduler.step()

        if isinstance(loss_fn, CombinedLoss):
            if loss_fn.current_epoch < loss_fn.warmup_epochs:
                phase_str = "[warmup]"
            elif loss_fn.current_epoch < loss_fn.warmup_epochs + loss_fn.transition_epochs:
                phase_str = "[transition]"
            else:
                phase_str = "[combined]"
        else:
            phase_str = ""
        print(f"Epoch {epoch+1}/{config.epochs} {phase_str} "
              f"Loss={avg_loss:.4f}, LR={current_lr:.2e}")

        # ==================== 定期评估 ====================
        is_eval_epoch = (epoch + 1) % config.eval_interval == 0 or epoch == 0
        is_full_eval = (epoch + 1) % config.full_eval_interval == 0

        if is_eval_epoch:
            # 使用 EMA 权重评估
            if ema is not None:
                ema.apply_shadow(model)

            # 快速评估 vs 全量评估
            use_fast = not is_full_eval
            avg_dice, avg_prec, avg_recall = evaluate(
                model, eval_dataset, config, device, epoch, fast=use_fast
            )

            if ema is not None:
                ema.restore(model)

            history['eval_epoch'].append(epoch + 1)
            history['eval_dice'].append(avg_dice)
            history['eval_precision'].append(avg_prec)
            history['eval_recall'].append(avg_recall)

            # ReduceOnPlateau 用 dice 做指标
            if config.scheduler == "reduce_on_plateau" and scheduler is not None:
                scheduler.step(avg_dice)

            # 保存最佳模型（只保存推理必需的内容，减小体积）
            if avg_dice > best_dice:
                best_dice = avg_dice
                no_improve_count = 0
                # 轻量版: 只保存推理所需
                best_path = os.path.join(config.save_dir, 'best_model.pth')
                raw_model = model.module if use_multi_gpu else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'ema_state_dict': {_strip_dp_k(k): v.cpu() for k, v in ema.shadow.items()} if ema else None,
                    'best_dice': best_dice,
                    'config': config.__dict__,
                }, best_path)
                print(f"  ★ 保存最佳模型: Dice={best_dice:.4f}")
            else:
                no_improve_count += 1  # 每次评估+1，不再加 eval_interval

            # 保存训练曲线
            draw_curves(history, config.log_dir)

        # ==================== 定期保存 checkpoint（防断网）====================
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(config.save_dir, 'latest_checkpoint.pth')
            raw_model = model.module if use_multi_gpu else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema_state_dict': {_strip_dp_k(k): v.cpu() for k, v in ema.shadow.items()} if ema else None,
                'best_dice': best_dice,
                'no_improve_count': no_improve_count,
                'history': history,
                'config': config.__dict__,
            }, ckpt_path)
            print(f"  💾 自动保存 checkpoint: Epoch {epoch + 1}")

        # ==================== 早停 ====================
        if no_improve_count >= config.patience:
            print(f"\n早停: 连续 {no_improve_count} 次评估无改善 "
                  f"(约 {no_improve_count * config.eval_interval} 个 epoch)")
            break

    # ==================== 训练结束 ====================
    # 保存最后模型
    last_path = os.path.join(config.save_dir, 'last_model.pth')
    _raw = model.module if use_multi_gpu else model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': _raw.state_dict(),
        'best_dice': best_dice,
        'config': config.__dict__,
    }, last_path)

    # 保存训练历史
    with open(os.path.join(config.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"训练完成！最佳 Dice = {best_dice:.4f}")
    print(f"最佳模型: {os.path.join(config.save_dir, 'best_model.pth')}")
    print(f"最后模型: {last_path}")
    print(f"训练曲线: {os.path.join(config.log_dir, 'training_curves.png')}")
    print("=" * 60)


if __name__ == "__main__":
    # 支持命令行参数覆盖，例如:
    # python train.py --epochs 100 --lr 1e-3 --model attention_unet
    try:
        config = get_config_from_args()
    except SystemExit:
        # 如果不带参数直接运行
        config = TrainConfig()

    train(config)
