"""
CTAI 时序预测模型
- TumorLSTM: 基于 LSTM 的肿瘤指标时序预测模型
- 用于根据患者历史诊断数据，预测未来肿瘤发展趋势
"""
import torch
import torch.nn as nn


class TumorLSTM(nn.Module):
    """
    基于 LSTM 的肿瘤指标时序预测模型

    输入: 历史特征序列 (batch, seq_len, input_size)
          input_size = 5 (面积、周长、灰度均值、灰度方差、似圆度)
    输出: 下一时间步的预测值 (batch, output_size)
          output_size = 5 (对应 5 个指标的预测)

    网络结构:
        输入层 → LayerNorm → LSTM(2层, 双向) → Dropout → FC → ReLU → FC → 输出
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                 output_size=5, dropout=0.2, bidirectional=True):
        super(TumorLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 输入归一化
        self.layer_norm = nn.LayerNorm(input_size)

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接输出层
        fc_input_size = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        返回: (batch, output_size) — 预测下一时间步的各指标值
        """
        # 输入归一化
        x = self.layer_norm(x)

        # LSTM 编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)

        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size * num_directions)

        # 全连接输出
        out = self.dropout(last_out)
        out = self.fc(out)  # (batch, output_size)

        return out

    def predict_multi_step(self, x, steps=3):
        """
        多步预测：迭代式预测未来 steps 个时间步

        x: (1, seq_len, input_size) — 历史序列
        steps: 预测步数
        返回: (steps, output_size) — 每步的预测值
        """
        self.eval()
        predictions = []

        current_input = x.clone()

        with torch.no_grad():
            for _ in range(steps):
                pred = self.forward(current_input)  # (1, output_size)
                predictions.append(pred.squeeze(0))

                # 将预测值加到序列末尾，用于下一步预测
                pred_step = pred.unsqueeze(1)  # (1, 1, output_size)
                current_input = torch.cat([current_input[:, 1:, :], pred_step], dim=1)

        return torch.stack(predictions, dim=0)  # (steps, output_size)
