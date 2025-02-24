import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torch.nn.functional as nnF
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision.models as models
from torchvision.models.resnet import conv1x1, conv3x3

from PIL import Image, ImageFilter
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from typing import Any
import shutil
import cv2

from typing import Optional, Callable
import multiprocessing

import sys
from loguru import logger as log
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import roc_curve, auc


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def get_time_for_name_some():
    return time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))


class EarlyStopping:
    def __init__(self, patience, delta, path, trace_func=print, mode="min_loss"):
        """
        :param patience: 等待多少个 epoch 没有提升后停止训练
        :param delta: 性能提升的最小变化，只有超过这个值才认为是提升
        :param path: 保存模型的文件路径
        :param trace_func: 记录函数，可以替换为 logger 等
        :param mode: 早停模式，'min_loss' 表示根据验证损失，'max_acc' 表示根据验证精度
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_loss = float("inf")
        self.best_val_acc = float("-inf")
        self.mode = mode

    def __call__(self, val_loss, val_acc, model):
        if self.mode == "min_loss":
            score = -val_loss
        else:  # mode == 'max_acc'
            score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
        elif (self.mode == "min_loss" and score < self.best_score + self.delta) or (
            self.mode == "max_acc" and score < self.best_score - self.delta * 100.0
        ):
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model):
        """当验证集的性能提升时保存模型"""
        if self.mode == "min_loss":
            self.trace_func(
                f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.best_val_loss = val_loss
        else:  # mode == 'max_acc'
            self.trace_func(
                f"Validation accuracy increased ({self.best_val_acc:.6f} --> {val_acc:.6f}).  Saving model ..."
            )
            self.best_val_acc = val_acc
        torch.save(model.state_dict(), self.path)


# 孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetwork, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层
        # ResNet50 的池化层是一个全局平均池化层，常用于提取图像的全局信息
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # 新增一个全连接层来进行二分类
        # 重新验证 先 batchnorm 再 Relu
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        # 提取 ResNet50 最后一个中间层的输出，[batch_size, 2048, 7, 7]
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        diff = torch.abs(output1 - output2)
        return self.classifier(diff), output1, output2


class SiameseNetwork2(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetwork2, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层
        # ResNet50 的池化层是一个全局平均池化层，常用于提取图像的全局信息
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        # 新增一个全连接层来进行二分类
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.fc1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        combined = torch.cat((output1, output2), dim=1)
        return self.classifier(combined)


class SiameseNetwork3(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetwork3, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层,池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # 新增一个全连接层来进行二分类
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        self.check(output1)
        self.check(output2)
        combined = torch.cat((output1, output2), dim=1)
        return self.classifier(combined)


class SiameseNetwork4(nn.Module):
    def __init__(
        self,
    ):
        super(SiameseNetwork4, self).__init__()

        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 移除 ResNet50 的全连接层,池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # 新增一个全连接层来进行二分类
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_params()

    def _init_params(self):
        # Initialize the weights for the fully connected layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan....")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        self.check(output1)
        self.check(output2)
        combined = torch.cat((output1, output2), dim=1)
        return self.classifier(combined)

# 自定义数据集，应用数据增强并选择负样本
class VtDataSet(Dataset):
    def __init__(
        self,
        dataset_dir,
        augmentations,
        preprocess,
        pos_net_probability=0.5,
    ):
        self.preprocess = preprocess
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations

        self.data_map = dict()
        self.neg_index_list = list()

        for data_path in os.listdir(dataset_dir):
            if not os.path.isfile(os.path.join(dataset_dir, data_path)):
                continue

            neg_dir_path = os.path.join(
                dataset_dir, os.path.splitext(data_path)[0], "approximate-negs"
            )

            self.data_map[os.path.join(self.dataset_dir, data_path)] = [
                os.path.join(
                    neg_dir_path,
                    neg_data_path,
                )
                for neg_data_path in os.listdir(neg_dir_path)
                if os.path.isfile(os.path.join(neg_dir_path, neg_data_path))
            ]

        self.dataset = list(self.data_map.keys())
        self.manager = multiprocessing.Manager()
        self.neg_index_list = self.manager.list([0] * len(self.dataset))

        self.pos_net_probability = pos_net_probability

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_img_path = self.dataset[index]
        # log.info(f"src_img_path: {src_img_path}")

        src_img = Image.open(src_img_path).convert("RGB")

        imgs = list()
        labels = list()
        process_src_img = self.preprocess(src_img)
        imgs.append(process_src_img)

        random_number = random.uniform(0, 1)
        # log.info(f"random_number: {random_number}, pos_net_probability: {self.pos_net_probability}")
        if random_number >= self.pos_net_probability:
            # pos
            imgs.append(self.augmentations(src_img))
            labels.append(1)
        else:
            neg_total_num = len(self.data_map[src_img_path])
            neg_index = random.randint(0, neg_total_num - 1)
            img = self.data_map[src_img_path][neg_index]
            neg_img = Image.open(img).convert("RGB")
            imgs.append(self.preprocess(neg_img))
            labels.append(0)

        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.float),
            self.dataset[index],
        )


def flatten_data(samples, labels, device):
    batch_size = samples.size(0)
    num_samples_per_image = samples.size(1)
    channels = samples.size(2)
    height = samples.size(3)
    width = samples.size(4)

    return samples.view(-1, channels, height, width).to(device), labels.view(-1).to(
        device
    )


# 训练函数
def train(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accumulation_steps,
    max_train_smaples_num,
):
    model.train()

    running_loss = 0.0

    train_count = 0.0

    total_len = len(loader) if max_train_smaples_num < 0 else max_train_smaples_num

    for i, (samples, labels, _) in enumerate(loader):
        if max_train_smaples_num > 0 and train_count >= max_train_smaples_num:
            log.info(f"train: {train_count} / {max_train_smaples_num}, stop train...\n")
            break

        train_refs = samples[:, 0, :, :, :].to(device)
        train_samples = samples[:, 1, :, :, :].to(device)
        train_labels = labels.to(device)

        optimizer.zero_grad()
        o1 = model(train_refs, train_samples)
        # 计算损失
        loss = criterion(o1, train_labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_count += 1.0

        batzh_size = samples.shape[0]
        log.info(
            f"train {train_count * batzh_size}/{total_len * batzh_size}:{((train_count ) / (total_len  ) * 100.0 ):.2f}% in one round...\n"
        )
        del train_refs, train_samples, train_labels, o1
        torch.cuda.empty_cache()

    return running_loss / train_count


# 验证函数
def validate(model, loader, device, criterion, max_val_smaples_num):
    model.eval()
    running_loss = 0.0
    valid_count = 0.0

    all_predicted_labels = list()
    all_predicted_output = list()
    all_labels = list()
    all_imgs = list()
    all_master_img_names = list()
    total_len = len(loader) if max_val_smaples_num < 0 else max_val_smaples_num

    with torch.no_grad():
        for i, (samples, labels, path) in enumerate(loader):
            if max_val_smaples_num > 0 and valid_count >= max_val_smaples_num:
                log.info(
                    f"valid: {valid_count} / {max_val_smaples_num}, stop valid...\n"
                )
                break

            valid_refs = samples[:, 0, :, :, :].to(device)
            valid_samples = samples[:, 1, :, :, :].to(device)
            valid_labels = labels.to(device)

            o1 = model(valid_refs, valid_samples)

            loss = criterion(o1, valid_labels)

            running_loss += loss.item()

            valid_count += 1
            if int(valid_count) % 100 == 0:
                log.info(
                    f"valid {valid_count}/{total_len}:{((valid_count ) / (total_len) * 100.0 ):.2f}% in one round...\n"
                )

            # 通过 sigmoid 将模型输出映射到 [0, 1] 区间
            probabilities = torch.sigmoid(o1.squeeze())
            # 四舍五入，计算预测标签
            preds = torch.round(probabilities).float()

            del valid_samples
            del valid_labels
            del o1
            torch.cuda.empty_cache()

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_names.append(file_name)
            flatten_samples, flatten_labels = flatten_data(samples, labels, "cpu")
            all_labels.append(flatten_labels.to("cpu").tolist())
            all_imgs.append(flatten_samples.to("cpu"))

            all_predicted_labels.append([preds.to("cpu").tolist()])
            all_predicted_output.append([probabilities.to("cpu").tolist()])

    return (
        running_loss / valid_count,
        all_predicted_output,
        all_predicted_labels,
        all_labels,
        all_imgs,
        all_master_img_names,
    )


def merge_images_with_text(
    master_img,
    master_img_denormalize,
    comparison_img,
    comparison_img_denormalize,
    predicted_output,
    predicted_label,
    label,
    is_correct,
    output_path,
    gap=10,
    title_height=30,
    side_padding=20,
    bottom_padding=20,
):
    # 可视化时 将标准化后负数的值进行
    # master_img[master_img < 0] = 0
    # comparison_img[comparison_img < 0] = 0
    # master_img_denormalize[master_img_denormalize < 0] = 0
    # comparison_img_denormalize[comparison_img_denormalize < 0] = 0

    # 将图像张量转换为 NumPy 数组
    master_img_denormalize_np = (
        master_img_denormalize.permute(1, 2, 0).cpu().numpy() * 255
    ).astype(int)
    comparison_img_denormalize_np = (
        comparison_img_denormalize.permute(1, 2, 0).cpu().numpy() * 255
    ).astype(int)

    # master_img_np[master_img_np < 0] = 0
    # master_img_denormalize_np[master_img_denormalize_np < 0] = 0
    # comparison_img_np[comparison_img_np < 0] = 0
    # comparison_img_denormalize_np[comparison_img_denormalize_np < 0] = 0

    # 创建左右两边的白色间隔
    left_right_padding = (
        np.ones((master_img_denormalize_np.shape[0], side_padding, 3), dtype=np.uint8)
        * 255
    )  # 白色间隔

    # 创建上下白色间隔
    top_bottom_padding_master = (
        np.ones(
            (bottom_padding, master_img_denormalize_np.shape[1] + 2 * side_padding, 3),
            dtype=np.uint8,
        )
        * 255
    )  # 白色间隔
    top_bottom_padding_comparison = (
        np.ones(
            (
                bottom_padding,
                comparison_img_denormalize_np.shape[1] + 2 * side_padding,
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )  # 白色间隔

    # 为每张图像添加左右白色间隔
    master_img_denormalize_padded = np.concatenate(
        (left_right_padding, master_img_denormalize_np, left_right_padding), axis=1
    )
    comparison_img_denormalize_padded = np.concatenate(
        (left_right_padding, comparison_img_denormalize_np, left_right_padding), axis=1
    )

    # 为每张图像添加上下白色间隔

    master_img_denormalize_padded = np.concatenate(
        (
            top_bottom_padding_master,
            master_img_denormalize_padded,
            top_bottom_padding_master,
        ),
        axis=0,
    )

    comparison_img_denormalize_padded = np.concatenate(
        (
            top_bottom_padding_comparison,
            comparison_img_denormalize_padded,
            top_bottom_padding_comparison,
        ),
        axis=0,
    )

    # 确定间隔的宽度和高度
    gap_array = (
        np.ones((master_img_denormalize_padded.shape[0], gap, 3), dtype=np.uint8) * 255
    )  # 白色间隔
    title_gap_array = (
        np.ones(
            (
                title_height,
                master_img_denormalize_padded.shape[1]
                + gap
                + comparison_img_denormalize_padded.shape[1]
                + gap,
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )  # 白色间隔

    # 创建一个新的图像，宽度是两张图像宽度之和加上间隔的宽度，高度是图像的高度加上标题高度
    merged_image = np.concatenate(
        (
            master_img_denormalize_padded,
            gap_array,
            comparison_img_denormalize_padded,
            gap_array,
        ),
        axis=1,
    )
    merged_image_with_title = np.concatenate((title_gap_array, merged_image), axis=0)

    # 创建一个图像窗口
    fig, ax = plt.subplots(figsize=(10, 5))

    # 显示合并后的图像
    ax.imshow(merged_image_with_title)

    # 在中央上方添加文字
    text = f"label: {bool(label)}, predicted label: {bool(predicted_label)}, predicted output: {predicted_output:.2f}, is_correct: {is_correct}, "
    text += f"\nLeft: Src img ---------- Right: Sample img"
    ax.text(
        merged_image_with_title.shape[1] // 2,
        15,
        text,
        fontsize=15,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # 去掉坐标轴
    ax.axis("off")

    # 保存合并后的图像
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Create a new tensor for denormalized values
    denormalized_tensor = tensor.clone()

    # Perform denormalization
    for t, m, s in zip(denormalized_tensor, mean, std):
        t.mul_(s).add_(m)

    return denormalized_tensor


def plot_distribution(data, filename):
    if len(data) == 0:
        return "None"

    # 2. 计算分布的度量
    mean = np.mean(data)
    median = np.median(data)
    mode = Counter(data).most_common(1)[0][0]
    variance = np.var(data)
    std_dev = np.std(data)

    msg = f"avg: {mean}, median: {median}, mode: {mode}, variance: {variance}, std_dev: {std_dev}, saved to {filename}"

    # 3. 绘制分布图
    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=100, kde=True)
    plt.title("Histogram")

    # 盒须图
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data)
    plt.title("Box and whisker plot")

    # 保存图像
    plt.savefig(filename)
    plt.close()
    return msg


def plot_roc_auc(
    y_true, y_scores, filename, title="Roc Plot (The closer auc is to 1, the better) "
):
    """
    绘制 ROC-AUC 曲线

    参数:
    y_true (array-like): 实际的二分类标签（0 或 1）
    y_scores (array-like): 模型的预测概率或决策函数的输出
    title (str): 图表标题（可选）

    返回:
    None
    """
    # 计算 ROC 曲线和 AUC 值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 计算 Youden's J statistic 的 阈值
    j_scores = tpr - fpr
    best_threshold_index = np.argmax(j_scores)
    best_threshold_by_youden = thresholds[best_threshold_index]

    # 计算距离 (0,1) 最近距离的阈值
    distances = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    best_threshold_index_distance = np.argmin(distances)
    best_threshold_by_distance = thresholds[best_threshold_index_distance]

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve: {roc_auc:.2f}")
    plt.scatter(
        fpr[best_threshold_index],
        tpr[best_threshold_index],
        marker="o",
        color="r",
        label=f"Best threshold by J: {best_threshold_by_youden:.2f}",
    )
    plt.scatter(
        fpr[best_threshold_index_distance],
        tpr[best_threshold_index_distance],
        marker="x",
        color="g",
        label=f"Best threshold by distance: {best_threshold_by_distance:.2f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(filename)
    plt.close()

    msg = f"roc_auc: {roc_auc}, Best threshold by Youdens J: {best_threshold_by_youden}, Best threshold by distance: {best_threshold_by_distance}, saved to {filename}"
    return msg


def plot_loss_curves(train_losses, val_losses, filename):
    """
    绘制训练损失和验证损失的变化曲线。

    参数:
    - train_losses (list of float): 训练损失列表
    - val_losses (list of float): 验证损失列表
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制训练损失曲线
    plt.plot(train_losses, label="Training Loss", color="blue", marker="o")

    # 绘制验证损失曲线
    plt.plot(val_losses, label="Validation Loss", color="red", marker="x")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def plot_acc_fp_fn_curves(acc_list, fp_list, fn_list, filename):
    """
    绘制训练损失和验证损失的变化曲线。

    参数:
    - train_losses (list of float): 训练损失列表
    - val_losses (list of float): 验证损失列表
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    plt.plot(
        acc_list,
        label="Acc",
        color="blue",
    )

    plt.plot(
        fp_list,
        label="FP",
        color="red",
    )

    plt.plot(fn_list, label="FN", color="green")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Acc && FP && FN")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def plot_recall_precision_curves(recall_list, precision_list, filename):
    """
    绘制训练损失和验证损失的变化曲线。

    参数:
    - train_losses (list of float): 训练损失列表
    - val_losses (list of float): 验证损失列表
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    plt.plot(
        recall_list,
        label="recall",
        color="blue",
    )

    plt.plot(
        precision_list,
        label="precision",
        color="red",
    )

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Recall && Precison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def calculat_correct(
    all_predicted_output,
    all_predicted_labels,
    all_labels,
    all_imgs,
    all_master_img_names,
    vt_mean,
    vt_std,
    output_plt_dir,
):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    num = 0.0

    tp_output_list = list()
    tn_output_list = list()
    fp_output_list = list()
    fn_output_list = list()

    for i in range(0, len(all_imgs)):
        master_img = all_imgs[i][0]
        master_img_denormalize = denormalize(master_img, vt_mean, vt_std)
        num += len(all_imgs[i]) - 1

        for j in range(0, len(all_labels[i])):
            label = all_labels[i][j]
            predicted_output = all_predicted_output[i][j]
            predicted_label = all_predicted_labels[i][j]
            sample_img = all_imgs[i][j + 1]
            sample_img_denormalize = denormalize(sample_img, vt_mean, vt_std)

            is_correct = False
            if label == 1:
                if predicted_label == 1:
                    is_correct = True
                    tp += 1
                    tp_output_list.append(predicted_output)

                else:
                    fn += 1
                    fn_output_list.append(predicted_output)
            else:
                if predicted_label == 0:
                    is_correct = True
                    tn += 1
                    tn_output_list.append(predicted_output)

                else:
                    fp += 1
                    fp_output_list.append(predicted_output)

            output_plt_path = os.path.join(
                output_plt_dir, f"{all_master_img_names[i]}-{j}-label{label}.png"
            )
            merge_images_with_text(
                master_img,
                master_img_denormalize,
                sample_img,
                sample_img_denormalize,
                predicted_output,
                predicted_label,
                label,
                is_correct,
                output_plt_path,
                50,
                100,
            )

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    log.info(f"this round have valid on {tp + fn} pos imgs and {tn + fp} neg imgs")

    return (
        recall * 100.0,
        precision * 100.0,
        (tp + tn) / num * 100.0,
        fp / num * 100.0,
        fn / num * 100.0,
        tp_output_list,
        tn_output_list,
        fp_output_list,
        fn_output_list,
    )


# 自定义转换：锐化
class Sharpen(transforms.Lambda):
    def __init__(self):
        super().__init__(self._apply_sharpen)

    def _apply_sharpen(self, img):
        return img.filter(ImageFilter.SHARPEN)


# 自定义转换：降噪（使用 OpenCV）
class RandomDenoise(transforms.Lambda):
    def __init__(self, h_range=(10, 30)):
        super().__init__(self._apply_denoise)
        self.h_range = h_range

    def _apply_denoise(self, img):
        img_np = np.array(img)
        # 从范围内随机选择降噪强度
        h = np.random.uniform(self.h_range[0], self.h_range[1])
        denoised_img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h, h, 7, 21)
        return Image.fromarray(denoised_img_np)


class CosineSimilarityBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityBCEWithLogitsLoss, self).__init__()

    def forward(self, input_vector, target_vector, target_labels):
        # 计算余弦相似度
        cosine_sim = nnF.cosine_similarity(input_vector, target_vector, dim=1)

        # 计算 BCEWithLogitsLoss
        loss = nnF.binary_cross_entropy_with_logits(cosine_sim, target_labels.float())
        return loss


def main():

    vt_mean = [0.485, 0.456, 0.406]
    vt_std = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose(
        [
            # 调整图像大小
            transforms.Resize(256),
            # 中心剪裁
            transforms.CenterCrop(224),
            # 转换为张量，并将像素值缩放到[0, 1]
            transforms.ToTensor(),
            # NormalizePerImage(),
            transforms.Normalize(mean=vt_mean, std=vt_std),
        ]
    )

    augmentations = transforms.Compose(
        [
            # 水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 旋转
            transforms.RandomRotation(degrees=15),
            # 先 resize
            transforms.Resize(256),
            # 随机剪裁
            transforms.RandomResizedCrop([224, 224], scale=(0.5, 1.0)),
            # 高斯模糊
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            # 亮度变化, 对比度，饱和度，色调
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=[0.0, 0.1]),  # type: ignore
            # h264 h265 常见锐化，降噪操作
            # 降噪
            RandomDenoise(h_range=(1, 10)),
            # 锐化
            Sharpen(),
            # 转换为张量，并将像素值缩放到[0, 1]
            transforms.ToTensor(),
            # NormalizePerImage(),
            transforms.Normalize(mean=vt_mean, std=vt_std),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Use {device}\n")

    gpu_count = torch.cuda.device_count()

    assets_base_dir_path = "/data/jinzijian/resnet50/assets"

    output_dir = f"/data/jinzijian/resnet50/output/{get_time_for_name_some()}"

    vt_train_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "vt-imgs-train"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    vt_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "vt-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    num_workers = 6

    total_batch_size = 256

    batch_size = 64

    # 梯度累积步数
    if gpu_count > 1:
        accumulation_steps = total_batch_size // (gpu_count * batch_size)
    else:
        accumulation_steps = total_batch_size // batch_size

    vt_train_loader = DataLoader(
        vt_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    vt_val_loader = DataLoader(
        vt_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    # 创建基于 ResNet50 的孪生网络
    # model = SiameseNetwork3(
    #     resetnet50_weight_path=os.path.join(
    #         assets_base_dir_path, "based-models", "gl18-tl-resnet50-gem-w-83fdc30.pth"
    #     )
    # )

    model = SiameseNetwork4()

    # 将模型迁移至多个 GPU 设备如果可行
    if gpu_count <= 1:
        model = model.to(device)
    else:
        model = nn.DataParallel(model)

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 设置学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

    # 设置损失函数
    criterion = nn.BCEWithLogitsLoss()

    # scaler = GradScaler()

    sum_message = ""

    best_model_wts = None
    best_acc = 0.0

    num_epochs = 300
    max_train_smaples_num = -1
    max_val_smaples_num = -1

    output_plt_dir = os.path.join(output_dir, "imgs")
    model_save_dir_path = os.path.join(output_dir, "model")

    log.info(f"all ouput will save to {output_dir}")

    try:
        shutil.rmtree(output_plt_dir)
    except Exception as e:
        pass

    os.makedirs(output_plt_dir, exist_ok=True)

    try:
        shutil.rmtree(model_save_dir_path)
    except Exception as e:
        pass

    os.makedirs(model_save_dir_path, exist_ok=True)
    model_save_file_path = os.path.join(
        model_save_dir_path, f"{get_time_for_name_some()}.pth"
    )

    data_distribution_plot_save_dir_path = os.path.join(output_dir, "data_distribution")
    try:
        shutil.rmtree(data_distribution_plot_save_dir_path)
    except Exception as e:
        pass

    os.makedirs(data_distribution_plot_save_dir_path, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=60,
        delta=0.01,
        path=model_save_file_path,
        mode="min_loss",
        trace_func=log.info,
    )

    train_loss_list = list()
    valid_loss_list = list()
    valid_acc_list = list()
    valid_fp_list = list()
    valid_fn_list = list()
    valid_recall_list = list()
    valid_precision_lits = list()

    for epoch in range(num_epochs):
        log.info(f"start train...\n")
        t1 = time.time()
        train_loss = train(
            model,
            vt_train_loader,
            criterion,
            optimizer,
            None,
            device,
            accumulation_steps,
            max_train_smaples_num,
        )
        t2 = time.time()
        # scheduler.step()

        log.info(f"start valid...\n")
        (
            valid_loss,
            all_predicted_output,
            all_predicted_labels,
            all_labels,
            all_imgs,
            all_master_img_names,
        ) = validate(model, vt_val_loader, device, criterion, max_val_smaples_num)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        epoch_output_plt_dir = os.path.join(output_plt_dir, str(epoch + 1))
        try:
            shutil.rmtree(epoch_output_plt_dir)
        except Exception as e:
            pass

        os.mkdir(epoch_output_plt_dir)
        log.info("start calculate acc/fp/fn...")
        (
            valid_recall,
            valid_precision,
            valid_acc,
            valid_fp,
            valid_fn,
            valid_tp_output_list,
            valid_tn_output_list,
            valid_fp_output_list,
            valid_fn_output_list,
        ) = calculat_correct(
            all_predicted_output,
            all_predicted_labels,
            all_labels,
            all_imgs,
            all_master_img_names,
            vt_mean,
            vt_std,
            epoch_output_plt_dir,
        )

        valid_acc_list.append(valid_acc)
        valid_fp_list.append(valid_fp)
        valid_fn_list.append(valid_fn)
        valid_recall_list.append(valid_recall)
        valid_precision_lits.append(valid_precision)

        epoch_data_distribution_plot_dir = os.path.join(
            data_distribution_plot_save_dir_path, str(epoch + 1)
        )
        try:
            shutil.rmtree(epoch_data_distribution_plot_dir)
        except Exception as e:
            pass

        os.mkdir(epoch_data_distribution_plot_dir)

        pos_file_name = os.path.join(epoch_data_distribution_plot_dir, "pos_plot.png")
        neg_file_name = os.path.join(epoch_data_distribution_plot_dir, "neg_plot.png")
        auc_file_name = os.path.join(epoch_data_distribution_plot_dir, "auc_plot.png")
        accfpfn_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "acc_fp_fn_plot.png"
        )
        recall_precision_name = os.path.join(
            epoch_data_distribution_plot_dir, "recall_precision_plot.png"
        )

        loss_curves_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "loss_curves.png"
        )

        tmp_msg = f"Device: {device}, Epoch {epoch+1}/{num_epochs}, "
        tmp_msg += f"Train cost: {(t2 - t1):.3f}s, Train loss: {train_loss:.4f}\n"
        tmp_msg += f"Valid loss: {valid_loss:.4f}, Recall: {valid_recall:.4f}%, Precision: {valid_precision:.4f}%\n"
        tmp_msg += (
            f"Valid acc: {valid_acc:.4f}%, fp: {valid_fp:.4f}%, fn: {valid_fn:.4f}%\n"
        )

        valid_pos_list = []
        valid_pos_list.extend(valid_tp_output_list)
        valid_pos_list.extend(valid_fn_output_list)

        valid_neg_list = []
        valid_neg_list.extend(valid_tn_output_list)
        valid_neg_list.extend(valid_fp_output_list)

        tmp_msg += f"Valid pos: {plot_distribution(valid_pos_list, pos_file_name)}\n"
        tmp_msg += f"Valid neg: {plot_distribution(valid_neg_list, neg_file_name)}\n"
        tmp_msg += f"Valid auc: {plot_roc_auc(all_labels, all_predicted_output, auc_file_name)}\n"
        tmp_msg += f"loss curves: {plot_loss_curves(train_loss_list, valid_loss_list, loss_curves_file_name)}\n"
        tmp_msg += f"Acc Fp Fn :{plot_acc_fp_fn_curves(valid_acc_list, valid_fp_list, valid_fn_list, accfpfn_file_name)}\n"
        tmp_msg += f"Rceall Presion :{plot_recall_precision_curves(valid_recall_list, valid_precision_lits, recall_precision_name)}\n"

        tmp_msg += "\n=====\n"
        sum_message += tmp_msg
        log.info(f"=== SUM === \n" + sum_message)

        early_stopping(model=model, val_loss=valid_loss, val_acc=valid_acc)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )
    main()
