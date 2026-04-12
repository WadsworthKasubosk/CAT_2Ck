import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class AttentionGate(nn.Module):
    """Attention Gate: 对 skip connection 特征加权，突出病灶区域"""
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: 来自解码器（gating signal）的通道数
        F_l: 来自编码器（skip connection）的通道数
        F_int: 中间特征的通道数
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUnet(nn.Module):
    """
    Attention U-Net + Deep Supervision + Bottleneck Dropout
    参考: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018

    新增特性:
    - deep_supervision=True 时，返回多尺度预测列表 [final, ds1, ds2, ds3]
    - Bottleneck Dropout 防止过拟合（37张极小数据集必备）
    - deep_supervision=False 时行为与原版完全一致（兼容已有权重）
    """
    def __init__(self, in_ch, out_ch, deep_supervision=False, dropout_rate=0.0):
        super(AttentionUnet, self).__init__()
        self.deep_supervision = deep_supervision

        # 编码器（与标准 U-Net 完全相同）
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        # Bottleneck Dropout（新增）
        self.bottleneck_dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # 解码器（与标准 U-Net 相同）
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        # Attention Gates（唯一新增的部分）
        self.att6 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att7 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att8 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att9 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Deep Supervision 输出头（新增）
        if deep_supervision:
            self.ds_conv_8x = nn.Conv2d(512, out_ch, 1)  # c6: 1/8 分辨率
            self.ds_conv_4x = nn.Conv2d(256, out_ch, 1)  # c7: 1/4 分辨率
            self.ds_conv_2x = nn.Conv2d(128, out_ch, 1)  # c8: 1/2 分辨率

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)

        # 编码器
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # Bottleneck Dropout
        c5 = self.bottleneck_dropout(c5)

        # 解码器 + Attention Gate
        up_6 = self.up6(c5)
        c4 = self.att6(g=up_6, x=c4)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c3 = self.att7(g=up_7, x=c3)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c2 = self.att8(g=up_8, x=c2)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c1 = self.att9(g=up_9, x=c1)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        # 最终输出（logits，不带 Sigmoid）
        final = self.conv10(c9)

        if self.deep_supervision and self.training:
            # 深监督：各层预测上采样到原始分辨率
            ds_8x = self.ds_conv_8x(c6)  # 1/8 分辨率
            ds_8x = nn.functional.interpolate(ds_8x, size=input_size, mode='bilinear', align_corners=False)

            ds_4x = self.ds_conv_4x(c7)  # 1/4 分辨率
            ds_4x = nn.functional.interpolate(ds_4x, size=input_size, mode='bilinear', align_corners=False)

            ds_2x = self.ds_conv_2x(c8)  # 1/2 分辨率
            ds_2x = nn.functional.interpolate(ds_2x, size=input_size, mode='bilinear', align_corners=False)

            # 返回: [最终预测, 1/8深监督, 1/4深监督, 1/2深监督]
            return [final, ds_8x, ds_4x, ds_2x]

        return final
