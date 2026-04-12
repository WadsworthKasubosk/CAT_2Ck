import sys
sys.path.append("..")

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader

from data_set import make
from net import attention_unet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)
torch.set_num_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("device:", device)

# =========================
# tiny 数据专用参数
# =========================
rate = 0.5
learn_rate = 1e-3
epochs = 50
batch_size = 2
pos_weight_value = 2.0
train_dataset_path = 'c:/Users/da983/CAT_2Ck/直肠癌数据_tiny/'
vis_dir = 'vis_results_tiny'

os.makedirs(vis_dir, exist_ok=True)

res = {
    'step': [],
    'loss': [],
    'dice': []
}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def dice_score_np(im1, im2):
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError(f"Shape mismatch: {im1.shape} vs {im2.shape}")

    if not (im1.any() or im2.any()):
        return 1.0

    intersection = np.logical_and(im1, im2)
    return 2.0 * intersection.sum() / (im1.sum() + im2.sum() + 1e-8)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous()
        targets = targets.contiguous()

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        return 1.0 - dice.mean()


def save_prediction_figure(image_np, target_np, prob_np, pred_bin, epoch, person_id, slice_id, save_dir):
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(target_np, cmap='gray')
    plt.title('GT Mask')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(prob_np, cmap='jet')
    plt.title('Pred Prob')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred_bin, cmap='gray')
    plt.title(f'Pred Binary@{rate}')
    plt.axis('off')

    plt.suptitle(f'Epoch {epoch + 1} | person={person_id} | slice={slice_id}')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'epoch_{epoch + 1}_sample.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"已保存可视化结果: {save_path}")


model = attention_unet.AttentionUnet(1, 1).to(device)
model.apply(weights_init)

train_dataset, test_dataset = make.get_d1(train_dataset_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

print("train size:", len(train_dataset))
print("test size:", len(test_dataset))

bce_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
)
dice_loss_fn = SoftDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


def compute_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    bce = bce_loss_fn(logits, targets)
    dice = dice_loss_fn(logits, targets)
    total = bce_weight * bce + dice_weight * dice
    return total, bce.item(), dice.item()


def evaluate(epoch):
    model.eval()
    total_dice = 0.0
    count = 0
    saved_vis = False

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x, y, person_id, slice_id = batch

            x = x.to(device)
            y = y.to(device)
            y = (y > 0).float()

            logits = model(x)
            probs = torch.sigmoid(logits)

            image_np = x.squeeze().cpu().numpy()
            target_np = y.squeeze().cpu().numpy()
            prob_np = probs.squeeze().cpu().numpy()
            pred_bin = (prob_np >= rate).astype(np.float32)
            target_bin = (target_np > 0).astype(np.float32)

            dice = dice_score_np(pred_bin, target_bin)
            total_dice += dice
            count += 1

            if idx == 0:
                print(f"[Eval Epoch {epoch + 1}] pred min/max: {prob_np.min():.6f}/{prob_np.max():.6f}")
                print(f"[Eval Epoch {epoch + 1}] pred mean   : {prob_np.mean():.6f}")
                print(f"[Eval Epoch {epoch + 1}] pred sum    : {pred_bin.sum()}")
                print(f"[Eval Epoch {epoch + 1}] target sum  : {target_bin.sum()}")
                print(f"[Eval Epoch {epoch + 1}] sample dice : {dice:.6f}")
                print(f"[Eval Epoch {epoch + 1}] person_id   : {person_id[0]}")
                print(f"[Eval Epoch {epoch + 1}] slice_id    : {slice_id[0]}")

            if not saved_vis:
                save_prediction_figure(
                    image_np=image_np,
                    target_np=target_np,
                    prob_np=prob_np,
                    pred_bin=pred_bin,
                    epoch=epoch,
                    person_id=person_id[0],
                    slice_id=slice_id[0],
                    save_dir=vis_dir
                )
                saved_vis = True

    avg_dice = total_dice / max(count, 1)
    print(f"Epoch {epoch + 1} Dice(on same tiny set): {avg_dice:.6f}")

    res['dice'].append(avg_dice)
    model.train()
    return avg_dice


def train():
    global_step = 0
    best_dice = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_dice_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            x, y, person_id, slice_id = batch

            x = x.to(device)
            y = y.to(device)
            y = (y > 0).float()

            optimizer.zero_grad()

            logits = model(x)
            loss, bce_value, dice_value = compute_loss(logits, y)

            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_total_loss += loss.item()
            epoch_bce_loss += bce_value
            epoch_dice_loss += dice_value

            res['step'].append(global_step)
            res['loss'].append(loss.item())

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Step {step}/{len(train_loader)} "
                f"Total Loss: {loss.item():.6f} "
                f"BCE: {bce_value:.6f} "
                f"DiceLoss: {dice_value:.6f}"
            )

        avg_total_loss = epoch_total_loss / max(len(train_loader), 1)
        avg_bce_loss = epoch_bce_loss / max(len(train_loader), 1)
        avg_dice_loss = epoch_dice_loss / max(len(train_loader), 1)

        print("=" * 60)
        print(f"Epoch {epoch + 1} Avg Total Loss: {avg_total_loss:.6f}")
        print(f"Epoch {epoch + 1} Avg BCE Loss  : {avg_bce_loss:.6f}")
        print(f"Epoch {epoch + 1} Avg Dice Loss : {avg_dice_loss:.6f}")
        print("=" * 60)

        avg_dice = evaluate(epoch)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), 'best_model_weights_tiny.pth')
            print(f"保存最佳模型: best_model_weights_tiny.pth, best dice = {best_dice:.6f}")

    torch.save(model.state_dict(), 'last_model_weights_tiny.pth')
    print("保存最后模型: last_model_weights_tiny.pth")

    draw_curve()
    print(f"训练完成，最佳 Dice(on same tiny set) = {best_dice:.6f}")


def draw_curve():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(res['step'], res['loss'], label='Train Loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(res['dice']) > 0:
        plt.plot(range(1, len(res['dice']) + 1), res['dice'],
                 label='Dice(on same tiny set)', color='#FF9966')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.title('Dice(on same tiny set)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("train_result_tiny.jpg", dpi=150)
    print("训练曲线已保存为 train_result_tiny.jpg")


if __name__ == '__main__':
    train()
