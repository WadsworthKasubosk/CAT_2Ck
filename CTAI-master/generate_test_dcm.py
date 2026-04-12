"""
生成一个可用的测试 DICOM 文件 (test_sample.dcm)
使用 SimpleITK 创建一张模拟 CT 图像的 512x512 灰度图
"""
import sys
import os
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("[ERROR] SimpleITK 未安装，正在安装...")
    os.system(f'"{sys.executable}" -m pip install SimpleITK')
    import SimpleITK as sitk

def generate_test_dcm(output_path="CTAI_flask/test_data/test_sample.dcm"):
    """生成一个带模拟肿瘤区域的测试 DICOM 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建 512x512 灰度图像（模拟 CT 切片）
    np.random.seed(42)
    size = 512

    # 背景：模拟人体组织的灰度值 (HU 大约 -100 ~ 100)
    image_array = np.random.normal(loc=80, scale=15, size=(1, size, size)).astype(np.int16)

    # 创建一个椭圆形的"器官"区域（灰度值稍高）
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    organ_mask = (x**2 / (150**2) + y**2 / (120**2)) <= 1
    image_array[0][organ_mask] += 30

    # 创建一个不规则的"肿瘤"区域（灰度值明显更高，模拟增强 CT 中的肿瘤）
    tumor_center_x, tumor_center_y = 30, -20
    tumor_mask = ((x - tumor_center_x)**2 / (35**2) + (y - tumor_center_y)**2 / (25**2)) <= 1
    image_array[0][tumor_mask] += 60

    # 在肿瘤中心加一个更亮的核心
    core_mask = ((x - tumor_center_x)**2 / (15**2) + (y - tumor_center_y)**2 / (10**2)) <= 1
    image_array[0][core_mask] += 30

    # 添加少量噪声让图像更自然
    noise = np.random.normal(0, 5, (size, size)).astype(np.int16)
    image_array[0] += noise

    # 限定灰度范围
    image_array = np.clip(image_array, 0, 4095)

    # 创建 SimpleITK Image
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing([0.7, 0.7, 5.0])
    sitk_image.SetOrigin([0.0, 0.0, 0.0])

    # 写入 DICOM
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.SetImageIO("GDCMImageIO")

    # 设置 DICOM 元数据
    sitk_image.SetMetaData("0008|0060", "CT")              # Modality
    sitk_image.SetMetaData("0008|0008", "ORIGINAL\\PRIMARY\\AXIAL")  # Image Type
    sitk_image.SetMetaData("0010|0010", "Test Patient")    # Patient Name
    sitk_image.SetMetaData("0010|0020", "TEST001")         # Patient ID
    sitk_image.SetMetaData("0010|0040", "M")               # Patient Sex
    sitk_image.SetMetaData("0010|1010", "050Y")            # Patient Age
    sitk_image.SetMetaData("0020|0013", "1")               # Instance Number
    sitk_image.SetMetaData("0028|0010", str(size))          # Rows
    sitk_image.SetMetaData("0028|0011", str(size))          # Columns
    sitk_image.SetMetaData("0028|0100", "16")               # Bits Allocated
    sitk_image.SetMetaData("0028|0101", "16")               # Bits Stored
    sitk_image.SetMetaData("0028|1052", "0")                # Rescale Intercept
    sitk_image.SetMetaData("0028|1053", "1")                # Rescale Slope

    writer.Execute(sitk_image)
    file_size = os.path.getsize(output_path)
    print(f"[OK] 测试 DCM 文件已生成: {output_path}")
    print(f"     文件大小: {file_size:,} 字节")
    print(f"     图像尺寸: {size}x{size}")
    print(f"     包含模拟肿瘤区域，可用于上传测试")

    # 同时复制一份到 uploads 目录方便测试
    uploads_path = "CTAI_flask/uploads/test_sample.dcm"
    os.makedirs(os.path.dirname(uploads_path), exist_ok=True)
    import shutil
    shutil.copy2(output_path, uploads_path)
    print(f"[OK] 已复制到: {uploads_path}")

    return output_path


if __name__ == "__main__":
    generate_test_dcm()
