import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),          # 将图像转换为Tensor
])

# 下载并加载 VOC 2012 训练集
voc_train = torchvision.datasets.VOCSegmentation(
    root='./VOC2012',  # 数据下载到的目录
    year='2012',    # 数据集年份
    image_set='train',  # 数据集类型：'train', 'val', 'trainval'
    download=True,  # 如果数据未下载，则下载数据
    transform=transform,  # 应用到图像的转换
    target_transform=transform  # 应用到标签的转换
)

# 定义 DataLoader
train_loader = DataLoader(voc_train, batch_size=16, shuffle=True, num_workers=2)

# 迭代数据集
for images, targets in train_loader:
    print(f'Images batch shape: {images.size()}')
    print(f'Targets batch shape: {targets.size()}')
    # 在这里进行训练或其他操作
    break  # 打印第一批数据后退出