# TEST 测试包说明

本目录为 CTAI 项目的测试资源打包目录。

## 目录结构

```
TEST/
├── dcm_samples/           # 测试用 DCM 文件
│   ├── 10013.dcm          # 真实直肠 CT 图像（520 KB）
│   └── test_sample.dcm    # 程序生成的模拟 CT（513 KB）
├── model_weights/         # 模型权重（需手动拷入，文件较大）
│   └── README.md          # 权重文件说明
├── screenshots/           # UI 功能测试截图
│   ├── fig01_system_homepage.png
│   ├── fig02_add_patient.png
│   ├── ...
│   └── fig13_time_series.png
└── README.md              # 本文件
```

## 测试 DCM 文件说明

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `10013.dcm` | 520 KB | 真实直肠 CT DICOM 影像，来自患者 1002 的动脉期扫描 |
| `test_sample.dcm` | 513 KB | `generate_test_dcm.py` 生成的模拟 CT 图像（几何图形） |

## 使用方法

1. 启动系统后，在前端选择患者
2. 点击「上传」按钮，选择 `dcm_samples/10013.dcm`
3. 等待模型推理完成（约 2-5 秒）
4. 查看原始 CT 图像和标注图

## 注意事项

- 真实 DCM 文件 `10013.dcm` 为脱敏医学影像数据
- `test_sample.dcm` 为程序自动生成，非真实医学影像，仅用于接口测试
