import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from PIL import Image
from torchvision import datasets

torch.manual_seed(12046)

# 将图片调整为28x28大小，转换为Tensor并归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 创建目标目录（train, val, test）
train_dir = './dataset/train'
val_dir = './dataset/val'
test_dir = './dataset/test'

# 创建目标目录
'''
for dir in [train_dir, val_dir, test_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
'''

def split_data(original_data_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    class_names = os.listdir(original_data_dir)

    for class_name in class_names:
        class_dir =os.path.join(original_data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 创建每个类别的训练、验证、测试子目录
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 获取当前类别下所有图像文件路径
        image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if
                       img.endswith(('jpg', 'png', 'jpeg'))]

        # 随机打乱图像路径
        random.shuffle(image_paths)

        # 计算每个子集的大小
        total_images = len(image_paths)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        test_size = total_images - train_size - val_size

        # 分割数据
        train_images = image_paths[:train_size]
        val_images = image_paths[train_size:train_size + val_size]
        test_images = image_paths[train_size + val_size:]

        # 将图像复制到相应的文件夹
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(train_dir, class_name, os.path.basename(img_path)))
        for img_path in val_images:
            shutil.copy(img_path, os.path.join(val_dir, class_name, os.path.basename(img_path)))
        for img_path in test_images:
            shutil.copy(img_path, os.path.join(test_dir, class_name, os.path.basename(img_path)))

#split_data(original_data_dir, train_dir, val_dir, test_dir)


train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=True)

print(f'Training set size: {len(train_dataset)}')
print(f'Validation set size: {len(val_dataset)}')
print(f'Test set size: {len(test_dataset)}')


eval_iters = 10

def estimate_loss(model):
    re = {}

    model.eval()
    re['train'] = _loss(model, train_loader)
    re['val'] = _loss(model, val_loader)
    re['test'] = _loss(model, test_loader)

    model.train()
    return re

@torch.no_grad()
def _loss(model, data_loader):
    loss = []
    accuracy = []
    data_iter = iter(data_loader)
    for k in range(eval_iters):
        inputs, labels = next(data_iter)
        B =inputs.shape[0]
        logits =model(inputs)
        loss.append(F.cross_entropy(logits, labels))
        _, predicted = torch.max(logits, 1)
        accuracy.append((predicted == labels).sum() / B)
    re = {
        'loss': torch.tensor(loss).mean().item(),
        'accuracy': torch.tensor(accuracy).mean().item()
    }
    return re

def train_cnn(model, optimizer, data_loader, epochs=10, penalty=[]):
    lossi = []
    for epoch in range(epochs):
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            lossi.append(loss.item())
            # 增加惩罚项
            for p in penalty:
                loss += p(model)
            loss.backward()
            optimizer.step()
        # 评估模型，并输出结果
        stats = estimate_loss(model)
        train_loss = f'train loss {stats["train"]["loss"]:.4f}'
        val_loss = f'val loss {stats["val"]["loss"]:.4f}'
        test_loss = f'test loss {stats["test"]["loss"]:.4f}'
        print(f'epoch {epoch:>2}: {train_loss}, {val_loss}, {test_loss}')
        train_acc = f'train acc {stats["train"]["accuracy"]:.4f}'
        val_acc = f'val acc {stats["val"]["accuracy"]:.4f}'
        test_acc = f'test acc {stats["test"]["accuracy"]:.4f}'
        print(f'{"":>10}{train_acc}, {val_acc}, {test_acc}')
    return lossi


stats = {}


# 在模型中加入批归一化层和随机失活
class CNN2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, (5, 5))
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 40, (5, 5))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40 * 4 * 4, 120)
        # 随机失活
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        B = x.shape[0]  # (B,  1, 28, 28)
        x = self.bn1(self.conv1(x))  # (B, 20, 24, 24)
        x = self.pool1(F.relu(x))  # (B, 20, 12, 12)
        x = self.bn2(self.conv2(x))  # (B, 40,  8,  8)
        x = self.pool2(F.relu(x))  # (B, 40,  4,  4)
        x = x.view(B, -1)  # (B, 40 * 4 * 4)
        x = F.relu(self.fc1(x))  # (B, 120)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 10)
        return x


model2 = CNN2()


stats['cnn2'] = train_cnn(model2, optim.Adam(model2.parameters(), lr=0.01), train_loader, epochs=10)

print(stats)