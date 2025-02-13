
import pickle
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


# 加载 CIFAR-10 数据的函数
def load_cifar10_batch(batch_file):
    with open(batch_file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'labels']

    # CIFAR-10 数据是 10000x3072 的 numpy 数组，需要 reshape 成 32x32x3
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为 (10000, 32, 32, 3)
    return images, labels


# 加载所有数据集
def load_cifar10_dataset(path):
    train_images = []
    train_labels = []

    # 加载五个训练批次
    for i in range(1, 6):
        batch_file = f'{path}/data_batch_{i}'
        images, labels = load_cifar10_batch(batch_file)
        train_images.append(images)
        train_labels.append(labels)

    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    # 加载测试集
    test_images, test_labels = load_cifar10_batch(f'{path}/test_batch')

    # 转换为 Hugging Face datasets 的格式
    train_dataset = Dataset.from_dict({"img": train_images, "label": train_labels})
    test_dataset = Dataset.from_dict({"img": test_images, "label": test_labels})

    return train_dataset, test_dataset


# 使用你的本地路径加载数据
def load_local_cifar10_as_dataset_dict(data_dir):
    train_dataset, test_dataset = load_cifar10_dataset(data_dir)

    dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    return dataset_dict

if __name__ == '__main__':
    dir = '/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/cifar10/cifar-10-batches-py'
    dataset_dict = load_local_cifar10_as_dataset_dict(dir)
    print(dataset_dict)

# # 应用预处理到数据集
# train_dataset = train_dataset.with_transform(transform)
# test_dataset = test_dataset.with_transform(transform)



