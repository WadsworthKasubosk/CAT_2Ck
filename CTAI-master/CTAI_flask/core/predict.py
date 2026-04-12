import os
import cv2
import torch
import core.net.unet as net
import numpy as np

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rate = 0.56


def predict(dataset, model):
    """
    dataset: (image_tensor_list, file_name)
    model: 已加载的 U-Net 模型
    """
    with torch.no_grad():
        x = dataset[0][0].to(device)       # [1, 1, H, W]
        file_name = dataset[1]
        y = model(x)                        # [1, 1, H, W]  带 Sigmoid 输出
        img_y = torch.squeeze(y).cpu().numpy()  # [H, W]
        img_y[img_y >= rate] = 255
        img_y[img_y < rate] = 0
        img_y = img_y.astype(np.uint8)
        cv2.imwrite(f'./tmp/mask/{file_name}_mask.png', img_y,
                    (cv2.IMWRITE_PNG_COMPRESSION, 0))
        print(f"[INFO] 推理完成: {file_name}, 预测前景像素数: {(img_y > 0).sum()}")
