from core import process, get_feature
from core.predict_yolo import predict_yolo


def c_main(path, model):
    """
    主推理链路（YOLO11-seg 版本）
    path: DCM 文件路径
    model: YOLO 模型对象
    """
    # 1. 预处理: DCM -> PNG
    yolo_input_path, file_name = process.pre_process(path)

    # 2. YOLO11-seg 推理 -> 生成 mask
    predict_yolo(yolo_input_path, file_name, model)

    # 3. 后处理: 原图 + mask -> 标注图
    process.last_process(file_name)

    # 4. 特征提取（不变，仍然从 mask 提取 24 项特征）
    image_info = get_feature.main(file_name)

    return file_name + '.png', image_info


if __name__ == '__main__':
    pass
