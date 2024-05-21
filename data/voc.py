import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # 载入图像文件名和标签文件名
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')
        self.image_list = os.path.join(self.root, 'ImageSets', 'Segmentation', f'{self.split}.txt')

        with open(self.image_list, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, f'{image_name}.jpg')
        label_path = os.path.join(self.label_dir, f'{image_name}.png')

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            label = np.array(label, dtype=np.int64)

        return image, torch.from_numpy(label)