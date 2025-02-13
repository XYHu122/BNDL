import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import datasets as DS
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
# import wandb
from utils.model_helpers import Proj_Model, KL_GamWei, batch_uncertain, uncertain_cal
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils.load_cifar100 import load_local_cifar100_as_dataset_dict
from utils.load_cifar10 import load_local_cifar10_as_dataset_dict
from utils.load_imagenet import load_local_imagenet_as_dataset_dict
import numpy as np
import os
import random
from argparse import ArgumentParser

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_distributed():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device

def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="imagenet-1k", help='dataset name')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    predata_fp = f"/mnt/wwn-0x5000c500d65d6ccc-part1/hxy_dataset/preprocessed_data/{args.dataset_name}"
    seed_everything(args.seed)

    # 加载数据集
    if dataset_name == 'cifar10':
        data_dir = '/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/cifar10/cifar-10-batches-py'
        dataset = load_local_cifar10_as_dataset_dict(data_dir)
        batch_size = 128
        num_labels = 10
    elif dataset_name == 'cifar100':
        data_dir = '/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/cifar100/cifar-100-python'
        dataset = load_local_cifar100_as_dataset_dict(data_dir)
        batch_size = 128
        num_labels = 100
    elif dataset_name == 'imagenet-1k':
        dir = "/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/DebuggableDeepNetworks/dataset/imagenet"
        dataset = load_local_imagenet_as_dataset_dict(dir)
        batch_size = 512
        num_labels = 1000
    else:
        raise NotImplementedError

    # 加载预训练的ViT模型和特征提取器
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224.model", num_labels=num_labels, ignore_mismatched_sizes=True)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224.processor")

    # 替换分类头
    feature_num, class_num = model.classifier.in_features, model.classifier.out_features
    model.classifier = Proj_Model(feature_num, class_num)

    # resume_fp = f"/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/Debuggable_ViT/saved_models/finetune_{dataset_name}_w0.02_new.pth"
    # model.load_state_dict(torch.load(resume_fp))

    # 冻结特征提取器的参数
    for param in model.vit.parameters():
        param.requires_grad = False

    model.vit.eval()  # 设置特征提取器为评估模式

    # 数据预处理函数
    if "cifar" in dataset_name:
        def transform_cifar(example_batch):
            imgs = [Image.fromarray(item) for item in np.array(example_batch['img'], dtype=np.uint8)]
            example_batch['pixel_values'] = processor(images=imgs, return_tensors='pt')['pixel_values']  # [0]
            return example_batch
        transform = transform_cifar
    else:
        # def transform_imagenet(examples):
        #     # 获取所有图像的路径
        #     img_paths = examples['img']
        #
        #     # 批量读取所有图像并转换为RGB格式
        #     imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
        #
        #     # 批量处理图像，使用 processor 进行预处理
        #     pixel_values = processor(images=imgs, return_tensors='pt')['pixel_values']
        #
        #     # 将处理后的 pixel_values 存入每个样本
        #     examples['pixel_values'] = pixel_values
        #
        #     return examples

        def transform_imagenet(example):
            img_paths = example['img']  # 从字典中获取图像路径
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths] #Image.open(img_path).convert("RGB")  # 打开图像并确保是 RGB 格式
            example['pixel_values'] = processor(images=imgs, return_tensors='pt')['pixel_values'] # processor(images=img, return_tensors='pt')['pixel_values'][0]
            return example
        transform = transform_imagenet

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        # print(f'pixel values {pixel_values}')
        labels = torch.tensor([item["label"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    # if os.path.exists(predata_fp):
    #     pre_dataset = DS.load_from_disk(predata_fp)
    # else:
    pre_dataset = dataset.with_transform(transform)
    # pre_dataset.save_to_disk(predata_fp)

    # 设置分布式训练
    device = setup_distributed()

    # 创建 DataLoader
    train_loader = DataLoader(
        pre_dataset['train'],
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=12,
        shuffle=True
    )
    test_loader = DataLoader(
        pre_dataset['test'],
        batch_size=batch_size,
        num_workers=12,
        collate_fn=collate_fn
    )

    # 将模型移动到设备并包装为 DDP
    model.to(device)
    model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    # 定义优化器和损失函数
    lr = 5e-5
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    config = {"learning_rate": lr,
              "epochs": num_epochs,
              "batch_size": batch_size, }
    # wandb.init(project="ViT_KL", name=f'{dataset_name}',  config=config)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if is_main_process():
            train_loader = tqdm(train_loader)
        for batch in train_loader:
            # 将输入数据移到GPU
            inputs = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            model_logits, z_out, weibull_lambda, k, weibull_lambda_w, k_w = outputs.logits

            recon_loss = criterion(model_logits, labels)

            # KL Loss
            gamma_shape = torch.tensor(1.0, dtype=torch.float32, requires_grad=False).to(device, non_blocking=True)
            gamma_scale = torch.tensor(1.0, dtype=torch.float32, requires_grad=False).to(device, non_blocking=True)
            KL = KL_GamWei(gamma_shape, gamma_scale, k, weibull_lambda)
            KL_w = KL_GamWei(gamma_shape, gamma_scale, k_w, weibull_lambda_w)

            loss = recon_loss + (KL + KL_w) * 1e-6

            # 反向传播
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = model_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if is_main_process():
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%")

        if (epoch + 1) % 1 == 0:
            # 测试循环
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                accurate_pred = torch.zeros([0], dtype=torch.float64).to(device, non_blocking=True)
                testresult = torch.zeros([0], dtype=torch.float64).to(device, non_blocking=True)

                if is_main_process():
                    test_loader = tqdm(test_loader)
                for batch in test_loader:
                    # 将输入数据移到GPU
                    inputs = batch["pixel_values"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    # 前向传播
                    outputs = model(inputs)
                    out, z_out, weibull_lambda, k, weibull_lambda_w, k_w = outputs.logits

                    # 统计正确预测的数量
                    _, predicted = out.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # uncertain_test
                    num_classes = class_num
                    model_logits, target, inp = out, labels, inputs
                    testresult, mean_logits, accurate_pred = batch_uncertain(model_logits, num_classes,
                                                                                 target, inp, model,
                                                                                 accurate_pred, testresult)
            if is_main_process():
                print(f"Epoch {epoch + 1} Test Accuracy: {100 * correct / total}%")
                # uncertain_test
                pavpus = uncertain_cal(testresult, mean_logits, accurate_pred)
                print(f'Epoch {epoch + 1} pavpus: {pavpus[0]:.4f}\t {pavpus[1]:.4f}\t {pavpus[2]:.4f}\t')
                # wandb.log(
                #     {
                #         "iter": epoch,
                #         "test_acc": f"{100 * correct / total}%",
                #         "pavpu_0.01": pavpus[0],
                #         "pavpu_0.05": pavpus[1],
                #         "pavpu_0.1": pavpus[2],
                #     }
                # )

            # 保存模型

            torch.save(model.module.state_dict(), f"/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/Debuggable_ViT/saved_models/finetune_{dataset_name}__w0.02_new_{epoch}ep.pth")
            # wandb.save(f"/mnt/wwn-0x5000c500e040f04e-part1/hxy/2024-XAI/Debuggable_ViT/saved_models/finetune_{dataset_name}.pth")

    cleanup_distributed()

if __name__ == "__main__":
    main()
