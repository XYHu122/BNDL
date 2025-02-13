import os
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision import io

with open("/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/Debuggable_ViT/utils/cleaned_win2id.txt", "r") as f:
    win2id ={}
    for line in f.readlines():
        id, win = line.split()
        win2id[win] = int(id)


def label_mapping(folder_win):
    return win2id[folder_win]

# 加载 ImageNet 数据的函数
def load_imagenet_dataset(path):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    label_map = {}

    # 加载train和val数据
    for split in ['train', 'val']:
        split_dir = os.path.join(path, split)
        for class_folder in tqdm(os.listdir(split_dir)):
            class_folder_path = os.path.join(split_dir, class_folder)
            if os.path.isdir(class_folder_path):
                if class_folder not in label_map:
                    label_counter = label_mapping(class_folder)
                    label_map[class_folder] = label_counter

                for image_file in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_file)
                    if split == 'train':
                        train_images.append(image_path)
                        train_labels.append(label_map[class_folder])
                    else:
                        val_images.append(image_path)
                        val_labels.append(label_map[class_folder])

    # 构建 Hugging Face datasets 格式的数据集
    train_dataset = Dataset.from_dict({"img": train_images, "label": train_labels})
    val_dataset = Dataset.from_dict({"img": val_images, "label": val_labels})

    return train_dataset, val_dataset

# 加载本地 ImageNet 数据集为 DatasetDict
def load_local_imagenet_as_dataset_dict(data_dir):
    train_dataset, val_dataset = load_imagenet_dataset(data_dir)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': val_dataset
    })
    return dataset_dict




# def transform_imagenet(example_batch):
#     # 定义预处理方法
#     preprocess = Compose([
#         Resize((224, 224)),  # ViT 模型通常需要 224x224 的输入
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])     # 处理每张图片
#     images = []
#     for img_path in example_batch['img']:
#         image = Image.open(img_path).convert("RGB")  # 确保图像为RGB格式
#         images.append(preprocess(image))      # 返回预处理后的图像
#         example_batch['pixel_values'] = images
#     return example_batch

if __name__ == '__main__':
    dir = "/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/imagenet"
    dataset_dict = load_local_imagenet_as_dataset_dict(dir)
    print(dataset_dict)