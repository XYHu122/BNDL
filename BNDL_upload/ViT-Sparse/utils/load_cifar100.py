import pickle
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


# 加载 CIFAR-100 数据的函数
def load_cifar100_batch(batch_file):
    with open(batch_file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'fine_labels']  # CIFAR-100 使用 'fine_labels' 表示 100 个类别的标签

    # CIFAR-100 数据是 10000x3072 的 numpy 数组，需要 reshape 成 32x32x3
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为 (10000, 32, 32, 3)
    return images, labels


# 加载所有数据集
def load_cifar100_dataset(path):
    # 加载训练批次
    train_file = f'{path}/train'
    train_images, train_labels = load_cifar100_batch(train_file)

    # 加载测试集
    test_file = f'{path}/test'
    test_images, test_labels = load_cifar100_batch(test_file)

    # 转换为 Hugging Face datasets 的格式
    train_dataset = Dataset.from_dict({"img": train_images, "label": train_labels})
    test_dataset = Dataset.from_dict({"img": test_images, "label": test_labels})

    return train_dataset, test_dataset


# 使用本地路径加载 CIFAR-100 数据集
def load_local_cifar100_as_dataset_dict(data_dir):
    train_dataset, test_dataset = load_cifar100_dataset(data_dir)

    dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    return dataset_dict

if __name__ == '__main__':
    dir = '/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/cifar100/cifar-100-python'
    dataset_dict = load_local_cifar100_as_dataset_dict(dir)
    print(dataset_dict)

