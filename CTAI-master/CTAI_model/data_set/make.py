import os
import SimpleITK as sitk
import cv2 as cv
import numpy as np
import torch
from torch.utils import data


def data_in_one(inputdata):
    inputdata = inputdata.astype(np.float32)
    if inputdata.max() == inputdata.min():
        return np.zeros_like(inputdata, dtype=np.float32)
    return (inputdata - inputdata.min()) / (inputdata.max() - inputdata.min())


def get_train_files(data_path):
    image_list = []

    dir_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    for dir_path in dir_list:
        phase_dir = os.path.join(dir_path, 'arterial phase')
        if not os.path.isdir(phase_dir):
            continue

        for name in os.listdir(phase_dir):
            full_path = os.path.join(phase_dir, name)
            if name.endswith('.dcm'):
                person_id = os.path.basename(dir_path)
                slice_id = name.replace('.dcm', '')
                image_list.append((full_path, person_id, slice_id))

    return image_list


class Dataset(data.Dataset):
    def __init__(self, path, have_mask=True):
        self.samples = []
        image_list = get_train_files(path)

        for dcm_path, person_id, slice_id in image_list:
            mask_path = dcm_path.replace('.dcm', '_mask.png')

            if not os.path.exists(mask_path):
                continue

            image = sitk.ReadImage(dcm_path)
            image_array = sitk.GetArrayFromImage(image)   # [1, H, W]

            if image_array.ndim != 3 or image_array.shape[0] != 1:
                continue

            image_array = data_in_one(image_array)

            mask_array = cv.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
            if mask_array is None:
                continue

            # 标准二值化
            mask_array = (mask_array > 0).astype(np.float32)

            # tiny 数据专用：只保留有前景的样本
            if have_mask and not mask_array.any():
                continue

            mask_array = np.expand_dims(mask_array, axis=0)  # [1, H, W]

            if image_array.shape != mask_array.shape:
                print(f"[跳过] shape 不匹配: {dcm_path}")
                print("image shape:", image_array.shape, "mask shape:", mask_array.shape)
                continue

            image_tensor = torch.from_numpy(image_array).float()
            mask_tensor = torch.from_numpy(mask_array).float()

            self.samples.append((image_tensor, mask_tensor, person_id, slice_id))

        print(f"加载完成，非空样本数: {len(self.samples)}")

    def __getitem__(self, index):
        image_tensor, mask_tensor, person_id, slice_id = self.samples[index]
        return image_tensor, mask_tensor, person_id, slice_id

    def __len__(self):
        return len(self.samples)


def get_d1(path):
    """
    tiny 数据专用：
    不切分 train/test，直接返回同一份数据做过拟合检查
    """
    bag = Dataset(path, have_mask=True)
    return bag, bag


if __name__ == '__main__':
    path = 'c:/Users/da983/CAT_2Ck/直肠癌数据_tiny/'
    train_dataset, test_dataset = get_d1(path)
    print("train:", len(train_dataset))
    print("test:", len(test_dataset))
