"""
CTAI 时序预测服务
- 集成 LSTM 模型预测 + 线性回归统计预测
- 被 app.py 调用，为趋势预测 API 提供数据支撑
"""
import os
import numpy as np
import torch

# =====================================================
# 线性回归预测（简单统计模型）
# =====================================================

def linear_regression_predict(values, predict_steps=3):
    """
    简单线性回归：用最小二乘法拟合历史数据，外推预测未来值

    参数:
        values: list[float|None] — 历史值序列（可能含 None）
        predict_steps: int — 预测步数

    返回: dict
        - predicted: list[float] — 预测的未来值
        - slope: float — 斜率（>0 趋势上升, <0 趋势下降）
        - intercept: float — 截距
        - r_squared: float — R² 拟合优度
        - trend: str — 趋势描述（增大/减小/稳定/数据不足）
        - confidence: str — 置信度描述
    """
    # 过滤 None 值，记录有效索引
    valid_pairs = [(i, v) for i, v in enumerate(values) if v is not None]

    if len(valid_pairs) < 2:
        return {
            'predicted': [],
            'slope': 0,
            'intercept': 0,
            'r_squared': 0,
            'trend': '数据不足',
            'confidence': '无法评估'
        }

    x = np.array([p[0] for p in valid_pairs], dtype=np.float64)
    y = np.array([p[1] for p in valid_pairs], dtype=np.float64)

    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    # 最小二乘法
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        slope = 0.0
        intercept = y_mean
    else:
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

    # R² 拟合优度
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 外推预测
    last_index = len(values) - 1
    predicted = []
    for step in range(1, predict_steps + 1):
        future_x = last_index + step
        pred_val = slope * future_x + intercept
        predicted.append(round(float(pred_val), 4))

    # 趋势判断（基于斜率的统计显著性）
    if len(valid_pairs) < 3:
        trend = '数据不足'
        confidence = '低'
    else:
        # 用变异系数判断斜率是否显著
        if y_mean != 0:
            relative_slope = abs(slope) / abs(y_mean)
        else:
            relative_slope = abs(slope)

        if relative_slope > 0.05:
            trend = '增大' if slope > 0 else '减小'
        else:
            trend = '稳定'

        if r_squared > 0.8:
            confidence = '高'
        elif r_squared > 0.5:
            confidence = '中'
        else:
            confidence = '低'

    return {
        'predicted': predicted,
        'slope': round(float(slope), 6),
        'intercept': round(float(intercept), 4),
        'r_squared': round(float(r_squared), 4),
        'trend': trend,
        'confidence': confidence
    }


# =====================================================
# LSTM 模型预测
# =====================================================

# 特征名列表，与 LSTM 模型的输入维度对应
FEATURE_NAMES = ['area', 'perimeter', 'mean', 'std', 'ellipse']
FEATURE_LABELS = {
    'area': '肿瘤面积',
    'perimeter': '肿瘤周长',
    'mean': '灰度均值',
    'std': '灰度方差',
    'ellipse': '似圆度'
}

# LSTM 模型全局变量
_lstm_model = None
_lstm_device = None
_normalizer = None  # 存储归一化参数


class FeatureNormalizer:
    """特征归一化器：Min-Max 归一化"""

    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.fitted = False

    def fit(self, data):
        """
        data: np.ndarray (n_samples, n_features)
        """
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)
        # 防止除零
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1.0
        self.fitted = True

    def transform(self, data):
        if not self.fitted:
            return data
        return (data - self.min_vals) / self.range_vals

    def inverse_transform(self, data):
        if not self.fitted:
            return data
        return data * self.range_vals + self.min_vals

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def _build_feature_matrix(area_values, perimeter_values, mean_values,
                          std_values, ellipse_values):
    """
    将各指标的时间序列组装为 LSTM 输入矩阵

    返回: np.ndarray (seq_len, 5)
    """
    n = len(area_values)
    matrix = np.zeros((n, 5), dtype=np.float64)

    for i in range(n):
        matrix[i, 0] = area_values[i] if area_values[i] is not None else 0
        matrix[i, 1] = perimeter_values[i] if perimeter_values[i] is not None else 0
        matrix[i, 2] = mean_values[i] if mean_values[i] is not None else 0
        matrix[i, 3] = std_values[i] if std_values[i] is not None else 0
        matrix[i, 4] = ellipse_values[i] if ellipse_values[i] is not None else 0

    return matrix


def _create_lstm_model():
    """
    创建 LSTM 模型实例（内联定义，避免跨目录导入问题）
    模型结构与 CTAI_model/net/time_series_model.py 中的 TumorLSTM 完全一致
    """
    import torch.nn as nn

    class TumorLSTM(nn.Module):
        def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                     output_size=5, dropout=0.2, bidirectional=True):
            super(TumorLSTM, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.layer_norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            fc_input_size = hidden_size * self.num_directions
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            x = self.layer_norm(x)
            lstm_out, (h_n, c_n) = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            out = self.dropout(last_out)
            out = self.fc(out)
            return out

        def predict_multi_step(self, x, steps=3):
            self.eval()
            predictions = []
            current_input = x.clone()
            with torch.no_grad():
                for _ in range(steps):
                    pred = self.forward(current_input)
                    predictions.append(pred.squeeze(0))
                    pred_step = pred.unsqueeze(1)
                    current_input = torch.cat([current_input[:, 1:, :], pred_step], dim=1)
            return torch.stack(predictions, dim=0)

    return TumorLSTM(
        input_size=5, hidden_size=64, num_layers=2,
        output_size=5, dropout=0.2, bidirectional=True
    )


def init_lstm_model(model_path=None):
    """
    初始化 LSTM 模型

    如果存在已训练的权重文件则加载，否则使用随机初始化的模型。
    对于数据量较小的场景，主要依赖在线微调（fine-tune）。
    """
    global _lstm_model, _lstm_device

    _lstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _lstm_model = _create_lstm_model().to(_lstm_device)

    # 尝试加载预训练权重
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'net', 'lstm_weights.pth'
        )

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=_lstm_device, weights_only=False)
            _lstm_model.load_state_dict(state_dict)
            print(f"[INFO] LSTM 时序模型加载成功: {model_path}")
        except Exception as e:
            print(f"[WARNING] LSTM 权重加载失败: {e}，使用随机初始化")
    else:
        print("[INFO] LSTM 暂无预训练权重，将使用在线微调模式")

    _lstm_model.eval()
    return _lstm_model


def online_finetune(feature_matrix, epochs=100, lr=0.01):
    """
    在线微调：用当前患者的历史数据快速训练 LSTM

    对于小数据集（< 50 个时间点），使用滑动窗口构建训练样本，
    用 MSE 损失进行快速微调。

    参数:
        feature_matrix: np.ndarray (seq_len, 5) — 归一化后的特征矩阵
        epochs: 微调轮次
        lr: 学习率
    """
    global _lstm_model, _lstm_device

    if _lstm_model is None:
        init_lstm_model()

    n = feature_matrix.shape[0]
    if n < 3:
        return  # 数据太少，无法微调

    # 滑动窗口构建训练样本
    # 窗口大小 = min(n-1, 5)
    window_size = min(n - 1, 5)

    X_train = []
    y_train = []

    for i in range(n - window_size):
        X_train.append(feature_matrix[i:i + window_size])
        y_train.append(feature_matrix[i + window_size])

    if len(X_train) == 0:
        return

    X_train = torch.FloatTensor(np.array(X_train)).to(_lstm_device)
    y_train = torch.FloatTensor(np.array(y_train)).to(_lstm_device)

    # 微调模型
    _lstm_model.train()
    optimizer = torch.optim.Adam(_lstm_model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = _lstm_model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    _lstm_model.eval()
    print(f"[INFO] LSTM 在线微调完成，最终 loss: {loss.item():.6f}")


def lstm_predict(area_values, perimeter_values, mean_values,
                 std_values, ellipse_values, predict_steps=3):
    """
    LSTM 时序预测主函数

    参数:
        area_values, ...: list[float|None] — 各指标的历史时间序列
        predict_steps: 预测步数

    返回: dict — 各指标的预测结果
    """
    global _lstm_model, _lstm_device, _normalizer

    # 在某些 Windows + Python 3.12 + torch CPU 环境下，LSTM forward 会触发段错误。
    # 设置环境变量 CTAI_DISABLE_LSTM=1 可禁用 LSTM 通路，仅走线性回归。
    if os.environ.get('CTAI_DISABLE_LSTM', '0') == '1':
        return {
            'success': False,
            'msg': 'LSTM 通路已禁用（CTAI_DISABLE_LSTM=1），仅使用线性回归预测',
            'predictions': {}
        }

    # 构建特征矩阵
    feature_matrix = _build_feature_matrix(
        area_values, perimeter_values, mean_values, std_values, ellipse_values
    )

    n = feature_matrix.shape[0]

    if n < 2:
        return {
            'success': False,
            'msg': '历史数据不足（至少需要 2 条记录）',
            'predictions': {}
        }

    # 初始化模型（如果尚未初始化）
    if _lstm_model is None:
        try:
            init_lstm_model()
        except Exception as e:
            return {
                'success': False,
                'msg': f'LSTM 模型初始化失败: {str(e)}',
                'predictions': {}
            }

    # 归一化
    _normalizer = FeatureNormalizer()
    normalized_matrix = _normalizer.fit_transform(feature_matrix)

    # 在线微调
    try:
        online_finetune(normalized_matrix, epochs=150, lr=0.005)
    except Exception as e:
        print(f"[WARNING] 在线微调失败: {e}")

    # 预测
    try:
        window_size = min(n, 5)
        input_seq = normalized_matrix[-window_size:]  # 取最后 window_size 个时间步

        input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(_lstm_device)

        with torch.no_grad():
            predictions = _lstm_model.predict_multi_step(input_tensor, steps=predict_steps)
            predictions = predictions.cpu().numpy()  # (predict_steps, 5)

        # 反归一化
        predictions = _normalizer.inverse_transform(predictions)

        # 组装结果
        result = {
            'success': True,
            'msg': 'LSTM 预测成功',
            'predictions': {}
        }

        for i, feat_name in enumerate(FEATURE_NAMES):
            pred_values = [round(float(v), 4) for v in predictions[:, i]]
            result['predictions'][feat_name] = {
                'name': FEATURE_LABELS.get(feat_name, feat_name),
                'predicted': pred_values
            }

        return result

    except Exception as e:
        return {
            'success': False,
            'msg': f'LSTM 预测失败: {str(e)}',
            'predictions': {}
        }


# =====================================================
# 综合预测接口（LSTM + 线性回归）
# =====================================================

def combined_predict(area_values, perimeter_values, mean_values,
                     std_values, ellipse_values, predict_steps=3):
    """
    综合预测接口：同时返回 LSTM 预测 和 线性回归预测，
    并给出综合评估结论。

    返回: dict
    """
    all_series = {
        'area': area_values,
        'perimeter': perimeter_values,
        'mean': mean_values,
        'std': std_values,
        'ellipse': ellipse_values,
    }

    # 1. 线性回归预测
    regression_results = {}
    for name, values in all_series.items():
        regression_results[name] = linear_regression_predict(values, predict_steps)
        regression_results[name]['name'] = FEATURE_LABELS.get(name, name)

    # 2. LSTM 预测
    lstm_results = lstm_predict(
        area_values, perimeter_values, mean_values,
        std_values, ellipse_values, predict_steps
    )

    # 3. 综合评估
    combined = {}
    for name in FEATURE_NAMES:
        reg = regression_results.get(name, {})
        lstm_pred = {}
        if lstm_results.get('success') and name in lstm_results.get('predictions', {}):
            lstm_pred = lstm_results['predictions'][name]

        # 综合预测值：如果 LSTM 成功，取两者平均；否则仅用线性回归
        if lstm_pred.get('predicted') and reg.get('predicted'):
            ensemble_predicted = []
            for lr_val, lstm_val in zip(reg['predicted'], lstm_pred['predicted']):
                # 加权平均：LSTM 60%, 线性回归 40%
                avg = round(0.6 * lstm_val + 0.4 * lr_val, 4)
                ensemble_predicted.append(avg)
        elif reg.get('predicted'):
            ensemble_predicted = reg['predicted']
        else:
            ensemble_predicted = lstm_pred.get('predicted', [])

        combined[name] = {
            'name': FEATURE_LABELS.get(name, name),
            'regression': reg,
            'lstm': lstm_pred,
            'ensemble_predicted': ensemble_predicted,
            'trend': reg.get('trend', '数据不足'),
            'confidence': reg.get('confidence', '无法评估'),
        }

    return {
        'lstm_success': lstm_results.get('success', False),
        'lstm_msg': lstm_results.get('msg', ''),
        'predict_steps': predict_steps,
        'features': combined
    }
