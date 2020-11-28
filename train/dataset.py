import os
import random
import torch.utils.data as data
import torch
import h5py
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    """
    dataroot: path of data file
    ratio: a tuple of length 2, e.g. (0, .5), means use [0, .5*total_img) as samples
    patch_size: default None(not crop).
    """

    def __init__(self, dataroot, type, i_fold=1, n_fold=2, argument=False,
                 patch_size=None, patch_interval=1, spectrum_limit=65535):
        super(dataset, self).__init__()
        assert type == 'train' or type == 'test' or type == 'all'
        rgb_file_name, spe_file_name = [], []
        self.argument = argument
        self.spectrum_limit = spectrum_limit
        for (roots, dirs, files) in os.walk(dataroot):
            if files == []: continue
            if "Thumbs.db" in files: files.remove("Thumbs.db")
            files.sort()
            rgb_file_name.append(roots + "/" + files[0])
            spe_file_name.append([roots + "/" + k for k in files[1:]])
        rgb_file_name.sort()
        spe_file_name.sort()
        img_length = len(rgb_file_name)
        assert img_length > 0
        if type != 'all':
            assert 1 <= i_fold <= n_fold
            test_radio = 1.0 / float(n_fold)
            test_st_pos = int((i_fold - 1) * test_radio * img_length)
            test_end_pos = int(((i_fold * test_radio) * img_length))
            if type == 'train':
                rgb_files = rgb_file_name[0:test_st_pos] + rgb_file_name[test_end_pos:img_length]
                spe_files = spe_file_name[0:test_st_pos] + spe_file_name[test_end_pos:img_length]
            else:
                rgb_files = rgb_file_name[test_st_pos:test_end_pos]
                spe_files = spe_file_name[test_st_pos:test_end_pos]
        else:
            rgb_files = rgb_file_name
            spe_files = spe_file_name
        self.rgb_names = rgb_files
        self.rgb = [transforms.ToPILImage()(self.bgr2rgb(cv2.imread(k)))
                    for k in rgb_files]
        self.spe = [[Image.open(k1) for k1 in k2] for k2 in spe_files]
        self.img_size = self.rgb[0].size[1]
        # for test
        self.patch_size = patch_size if patch_size is not None else self.img_size
        assert isinstance(self.patch_size, int)
        self.patch_interval = patch_interval
        self.dim_per_image = (self.img_size - self.patch_size) // self.patch_interval \
                             + (1 if self.img_size >= self.patch_size else 0)
        self.patch_per_image = self.dim_per_image ** 2
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_idx = idx // self.patch_per_image
        img = [self.rgb[img_idx]] + self.spe[img_idx]
        if self.patch_per_image != 1:
            patch_idx = idx % self.patch_per_image
            col_idx = patch_idx // self.dim_per_image * self.patch_interval
            row_idx = patch_idx % self.dim_per_image * self.patch_interval
            # Data Augmentation
            if self.argument:
                row_idx_range = range(max(0, row_idx - self.patch_interval + 1),
                                      row_idx + 1)
                col_idx_range = range(max(0, col_idx - self.patch_interval + 1),
                                      col_idx + 1)
                row_idx = random.choice(row_idx_range)
                col_idx = random.choice(col_idx_range)
            for k in range(32):
                img[k] = img[k].crop((row_idx,
                                      col_idx,
                                      row_idx + self.patch_size,
                                      col_idx + self.patch_size))
        if self.argument:
            seed_1 = random.random()
            seed_2 = random.random()
            for k in range(32):
                img[k] = self.data_argument(img[k], seed_1, seed_2)
        # for rgb img
        # To Tensor, from HWC to CHW format
        img[0] = self.transform(img[0])
        # for spe img
        for k in range(1, 32):
            img_arr = np.array(img[k], np.int32, copy=True)
            img_arr = torch.from_numpy(img_arr)
            img_arr = img_arr.view(img[k].size[1], img[k].size[0], len(img[k].getbands()))
            # put it from HWC to CHW format
            img_arr = img_arr.permute((2, 0, 1)).contiguous()
            img[k] = img_arr.float().div(self.spectrum_limit)

        rgb, spe = img[0], torch.cat(img[1:], dim=0)
        return rgb, spe

    def __len__(self):
        return len(self.rgb) * self.patch_per_image

    def get_dataloader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(dataset=self, shuffle=shuffle,
                                           batch_size=batch_size, num_workers=0)

    def bgr2rgb(self, mat):
        return cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    def data_argument(self, img, seed1, seed2):
        if seed1 > 0.5: img = TF.vflip(img)
        if seed2 > 0.5: img = TF.hflip(img)
        return img


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        print(hf.keys())
        self.data = hf.get('data')
        self.target = hf.get('label')
        print(type(self.data))

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :]).float(), torch.from_numpy(
            self.target[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    ds = dataset('./dataset/CAVE', (0, 0.5), patch_size=64, patch_interval=128)
    for i in range(16):
        rgb, hhs = ds[i]
        transforms.ToPILImage()(rgb).show()
