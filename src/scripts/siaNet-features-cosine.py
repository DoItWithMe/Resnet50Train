import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torch.nn.functional as nnF

import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image, ImageFilter
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import shutil
import cv2

import multiprocessing

import sys
from loguru import logger as log
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import roc_curve, auc
from torch.nn.parameter import Parameter


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


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        p: int = 3
        eps: float = 1e-6
        input = x.clamp(min=eps)
        _input = input.pow(p)
        t = nnF.avg_pool2d(_input, (7, 7)).pow(1.0 / p)

        return t

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class SiameseNetworkCosineNet1(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetworkCosineNet1, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层 和 池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        # self.pool = nn.MaxPool2d(kernel_size=7)

        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
        x = nnF.normalize(x)
        return x

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan...., {output}")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3


class SiameseNetworkCosineNet2(nn.Module):
    def __init__(self):
        super(SiameseNetworkCosineNet2, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 移除 ResNet50 的全连接层 和 池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
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

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan...., {output}")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3


class SiameseNetworkCosineNet3(nn.Module):
    def __init__(self):
        super(SiameseNetworkCosineNet3, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        )

        # 移除 ResNet50 的全连接层 和 池化层
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        self.fc1 = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
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

    def check(self, output):
        if torch.isnan(output).any():
            log.warning(f"output have nan...., {output}")

        if torch.isinf(output).any() > 0:
            log.warning(f"output have inf....")

    def forward(self, input1, input2, input3):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        output3 = self.forward_one(input3)

        self.check(output1)
        self.check(output2)
        self.check(output3)

        return output1, output2, output3


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

        # random_number = random.uniform(0, 1)
        # if random_number >= self.pos_net_probability:
        imgs.append(self.augmentations(src_img))
        labels.append(1)
        # else:
        neg_total_num = len(self.data_map[src_img_path])
        neg_index = random.randint(0, neg_total_num - 1)
        img = self.data_map[src_img_path][neg_index]
        neg_img = Image.open(img).convert("RGB")
        imgs.append(self.preprocess(neg_img))
        labels.append(-1)

        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.float),
            self.dataset[index],
        )


class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor_normal = nnF.normalize(anchor)
        positive_normal = nnF.normalize(positive)
        negative_normal = nnF.normalize(negative)

        # 计算余弦相似度
        pos_sim = nnF.cosine_similarity(anchor_normal, positive_normal, dim=1)
        neg_sim = nnF.cosine_similarity(anchor_normal, negative_normal, dim=1)

        pos_dis = 1 - pos_sim
        neg_dis = 1 - neg_sim

        # 计算损失
        # from ChatGpt
        # loss = nnF.relu(neg_sim - pos_sim + self.margin).mean()
        loss = nnF.relu(pos_dis - neg_dis + self.margin).mean()
        return loss


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
        train_pos_samples = samples[:, 1, :, :, :].to(device)
        train_neg_samples = samples[:, 2, :, :, :].to(device)
        train_labels = labels.to(device)

        optimizer.zero_grad()
        o1, o2, o3 = model(train_refs, train_pos_samples, train_neg_samples)
        # 计算损失
        # log.info(f"o1: {o1.shape}, train_labels: {train_labels.squeeze(1).shape}")
        loss = criterion(o1, o2, o3)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_count += 1.0

        batzh_size = samples.shape[0]
        log.info(
            f"train {train_count * batzh_size}/{total_len * batzh_size}:{((train_count ) / (total_len  ) * 100.0 ):.2f}% in one round...\n"
        )
        del train_refs, train_pos_samples, train_neg_samples, train_labels
        torch.cuda.empty_cache()

    return running_loss / train_count


# 验证函数
def validate(model, loader, device, criterion, max_val_smaples_num):
    model.eval()
    running_loss = 0.0
    valid_count = 0.0

    all_labels = list()
    all_imgs = list()
    all_master_img_names = list()
    all_predicted_cosine_sim = list()

    total_len = len(loader) if max_val_smaples_num < 0 else max_val_smaples_num

    with torch.no_grad():
        for i, (samples, labels, path) in enumerate(loader):
            if max_val_smaples_num > 0 and valid_count >= max_val_smaples_num:
                log.info(
                    f"valid: {valid_count} / {max_val_smaples_num}, stop valid...\n"
                )
                break

            valid_refs = samples[:, 0, :, :, :].to(device)
            valid_pos_samples = samples[:, 1, :, :, :].to(device)
            valid_neg_samples = samples[:, 2, :, :, :].to(device)
            valid_labels = labels.to(device)

            o1, o2, o3 = model(valid_refs, valid_pos_samples, valid_neg_samples)

            loss = criterion(o1, o2, o3)

            running_loss += loss.item()

            valid_count += 1
            if int(valid_count) % 100 == 0:
                log.info(
                    f"valid {valid_count}/{total_len}:{((valid_count ) / (total_len) * 100.0 ):.2f}% in one round...\n"
                )

            # cosine_similarity value range [-1, 1],
            # change it to [0, 2]
            pos_cosine_sim = (
                nnF.cosine_similarity(nnF.normalize(o1), nnF.normalize(o2), dim=1) + 1
            )
            neg_cosine_sim = (
                nnF.cosine_similarity(nnF.normalize(o1), nnF.normalize(o3), dim=1) + 1
            )

            del valid_pos_samples, valid_neg_samples
            del valid_labels
            del o1, o2, o3
            torch.cuda.empty_cache()

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_names.append(file_name)
            flatten_samples, flatten_labels = flatten_data(samples, labels, "cpu")

            all_imgs.append(flatten_samples.to("cpu"))
            all_labels.append([flatten_labels.to("cpu").tolist()[0]])
            all_labels.append([flatten_labels.to("cpu").tolist()[1]])
            all_predicted_cosine_sim.append([pos_cosine_sim.to("cpu").tolist()[0]])
            all_predicted_cosine_sim.append([neg_cosine_sim.to("cpu").tolist()[0]])

    return (
        running_loss / valid_count,
        all_labels,
        all_imgs,
        all_master_img_names,
        all_predicted_cosine_sim,
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
    # 1. 统计各个数字的频率

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

    # 计算 Youden's J statistic
    j_scores = tpr - fpr
    best_threshold_index = np.argmax(j_scores)
    best_threshold_youden = thresholds[best_threshold_index]
    # 计算距离 (0,1) 点的距离
    distances = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    best_threshold_index_distance = np.argmin(distances)
    best_threshold_distance = thresholds[best_threshold_index_distance]

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve: {roc_auc:.2f}")
    plt.scatter(
        fpr[best_threshold_index],
        tpr[best_threshold_index],
        marker="o",
        color="r",
        label=f"Best threshold by J: {best_threshold_youden:.4f}",
    )
    plt.scatter(
        fpr[best_threshold_index_distance],
        tpr[best_threshold_index_distance],
        marker="x",
        color="g",
        label=f"Best threshold by distance: {best_threshold_distance:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(filename)
    plt.close()

    msg = f"roc_auc: {roc_auc}, Best threshold by Youdens J: {best_threshold_youden}, Best threshold by distance: {best_threshold_distance}, saved to {filename}"
    return msg, best_threshold_youden, best_threshold_distance


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


def plot_acc_fp_fn_curves(acc_list, fp_list, fn_list, cosine_threshold_list, filename):
    """
    绘制训练损失和验证损失的变化曲线。

    参数:
    - train_losses (list of float): 训练损失列表
    - val_losses (list of float): 验证损失列表
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    if cosine_threshold_list is not None:
        plt.plot(cosine_threshold_list, acc_list, label="Acc", color="blue", marker="x")

        plt.plot(cosine_threshold_list, fp_list, label="FP", color="red", marker="o")

        plt.plot(cosine_threshold_list, fn_list, label="FN", color="green", marker="v")
    else:
        plt.plot(acc_list, label="Acc", color="blue", marker="x")

        plt.plot(fp_list, label="FP", color="red", marker="o")

        plt.plot(fn_list, label="FN", color="green", marker="v")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Acc && FP && FN")
    plt.ylabel("Percentage")
    if cosine_threshold_list is not None:
        plt.xlabel("Threshold")
        plt.xticks(
            ticks=cosine_threshold_list,
            labels=[f"{x:.2f}" for x in cosine_threshold_list],
            rotation=45,
        )
    else:
        plt.xlabel("Round")

    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def plot_recall_precision_curves(
    recall_list, precision_list, cosine_threshold_list, filename
):
    """
    绘制召回率(Recall)和精确率(Precision)的变化曲线,并在 X 轴上显示对应的阈值。

    参数:
    - recall_list (list of float): 召回率列表
    - precision_list (list of float): 精确率列表
    - threshold_list (list of float): L2 距离阈值列表，对应 X 轴刻度
    - filename (str): 图像保存路径
    """  # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    if cosine_threshold_list is not None:
        plt.plot(
            cosine_threshold_list, recall_list, label="recall", color="blue", marker="x"
        )

        plt.plot(
            cosine_threshold_list,
            precision_list,
            label="precision",
            color="red",
            marker="o",
        )
    else:
        plt.plot(recall_list, label="recall", color="blue", marker="x")

        plt.plot(
            precision_list,
            label="precision",
            color="red",
            marker="o",
        )

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Recall && Precison")
    plt.ylabel("Percentage")

    if cosine_threshold_list is not None:
        plt.xlabel("Threshold")
        plt.xticks(
            ticks=cosine_threshold_list,
            labels=[f"{x:.2f}" for x in cosine_threshold_list],
            rotation=45,
        )
    else:
        plt.xlabel("Round")

    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def calculat_correct_with_threshold(
    all_labels,
    all_imgs,
    all_master_img_names,
    vt_mean,
    vt_std,
    output_plt_dir,
    all_predicted_cosine_sim,
    cosine_threshold_list,
):  # -> tuple[list[float], list[float], list[float], list[float],...:

    num = 0.0

    tp_list = [0 for i in range(len(cosine_threshold_list))]
    tn_list = [0 for i in range(len(cosine_threshold_list))]
    fp_list = [0 for i in range(len(cosine_threshold_list))]
    fn_list = [0 for i in range(len(cosine_threshold_list))]

    fpr_list = [0.0 for i in range(len(cosine_threshold_list))]
    fnr_list = [0.0 for i in range(len(cosine_threshold_list))]
    acc_list = [0.0 for i in range(len(cosine_threshold_list))]
    recall_list = [0.0 for i in range(len(cosine_threshold_list))]
    precision_list = [0.0 for i in range(len(cosine_threshold_list))]

    neg_cosine_sim_list = list()
    pos_cosine_sim_list = list()

    for i in range(0, len(all_imgs)):
        master_img = all_imgs[i][0]
        master_img_denormalize = denormalize(master_img, vt_mean, vt_std)
        num += len(all_imgs[i]) - 1

        for j in range(2 * i, 2 * (i + 1)):
            img_index = j - 2 * i
            sample_img = all_imgs[i][img_index]
            sample_img_denormalize = denormalize(sample_img, vt_mean, vt_std)

            cosine_sim = all_predicted_cosine_sim[j][0]
            label = all_labels[j][0]

            output_plt_path = os.path.join(
                output_plt_dir, f"{all_master_img_names[i]}-{j}-label{label}.png"
            )
            merge_images_with_text(
                master_img,
                master_img_denormalize,
                sample_img,
                sample_img_denormalize,
                cosine_sim,
                -999,
                label,
                False,
                output_plt_path,
                50,
                100,
            )

            if label == 1:
                pos_cosine_sim_list.append(cosine_sim)
            else:
                neg_cosine_sim_list.append(cosine_sim)

            for k, cosine_threshold in enumerate(cosine_threshold_list):
                predicted_label = 1 if cosine_sim >= cosine_threshold else 0
                if label == 1:
                    if predicted_label == 1:
                        tp_list[k] += 1
                    else:
                        fn_list[k] += 1
                else:
                    if predicted_label == 0:
                        tn_list[k] += 1
                    else:
                        fp_list[k] += 1

    for i in range(len(tp_list)):
        tp = tp_list[i]
        fn = fn_list[i]
        tn = tn_list[i]
        fp = fp_list[i]

        acc_list[i] = (tp + tn) / num * 100.0
        fpr_list[i] = fp / num * 100.0
        fnr_list[i] = fn / num * 100.0
        recall_list[i] = (tp / (tp + fn) if (tp + fn) > 0 else 0.0) * 100.0
        precision_list[i] = (tp / (tp + fp) if (tp + fp) > 0 else 0.0) * 100.0

    return (
        acc_list,
        fpr_list,
        fnr_list,
        recall_list,
        precision_list,
        pos_cosine_sim_list,
        neg_cosine_sim_list,
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


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / epochs))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


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

    batch_size = 40

    # 梯度累积步数
    if gpu_count > 1:
        accumulation_steps = total_batch_size // (gpu_count * batch_size)
    else:
        accumulation_steps = total_batch_size // batch_size

    vt_train_loader = DataLoader(
        vt_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    vt_val_loader = DataLoader(
        vt_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    # 创建基于 ResNet50 的孪生网络
    # model = SiameseNetworkCosineNet1(
    #     resetnet50_weight_path=os.path.join(
    #         assets_base_dir_path, "based-models", "gl18-tl-resnet50-gem-w-83fdc30.pth"
    #     )
    # )
    model = SiameseNetworkCosineNet3()

    # 将模型迁移至多个 GPU 设备如果可行
    if gpu_count <= 1:
        model = model.to(device)
    else:
        model = nn.DataParallel(model)

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # init_lr = batch_size / 256
    # optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=0.9)

    # 设置学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

    # 设置损失函数
    # criterion = nn.CosineEmbeddingLoss(0.5)
    criterion = TripletCosineLoss(1)

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

    valid_acc_list_by_youden = list()
    valid_acc_list_by_dis = list()

    valid_fp_list_by_youden = list()
    valid_fp_list_by_dis = list()

    valid_fn_list_by_youden = list()
    valid_fn_list_by_dis = list()

    valid_recall_list_by_youden = list()
    valid_recall_list_by_dis = list()

    valid_precision_list_by_youden = list()
    valid_precision_list_by_dis = list()

    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, init_lr, epoch, num_epochs)
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

        log.info(f"start valid...\n")
        (
            valid_loss,
            all_labels,
            all_imgs,
            all_master_img_names,
            all_predicted_cosine_sim,
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

        epoch_data_distribution_plot_dir = os.path.join(
            data_distribution_plot_save_dir_path, str(epoch + 1)
        )
        try:
            shutil.rmtree(epoch_data_distribution_plot_dir)
        except Exception as e:
            pass

        os.mkdir(epoch_data_distribution_plot_dir)

        loss_curves_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "loss_curves.png"
        )

        pos_dis_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "pos_dis_plot.png"
        )

        neg_dis_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "neg_dis_plot.png"
        )

        acc_fp_fn_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "acc_fpr_fnr_plot.png"
        )

        recall_precision_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "recall_presion_plot.png"
        )

        acc_fp_fn_by_youden_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "youden_acc_fpr_fnr_plot.png"
        )

        recall_precision_by_youden_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "youden_recall_presion_plot.png"
        )

        acc_fp_fn_by_dis_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "dis_acc_fpr_fnr_plot.png"
        )

        recall_precision_by_dis_file_name = os.path.join(
            epoch_data_distribution_plot_dir, "dis_recall_presion_plot.png"
        )

        auc_file_name = os.path.join(epoch_data_distribution_plot_dir, "auc_plot.png")

        tmp_msg = f"Device: {device}, Epoch {epoch+1}/{num_epochs}, "
        tmp_msg += f"Train cost: {(t2 - t1):.3f}s, Train loss: {train_loss:.4f} Valid loss: {valid_loss:.4f}\n"
        tmp_msg += f"loss curves: {plot_loss_curves(train_loss_list, valid_loss_list, loss_curves_file_name)}\n"

        tmsg, best_threshold_youden, best_threshold_distance = plot_roc_auc(
            all_labels, all_predicted_cosine_sim, auc_file_name
        )
        tmp_msg += f"{tmsg}\n"

        cosine_threshold_list = [best_threshold_youden, best_threshold_distance]
        (
            valid_acc_list,
            valid_fpr_list,
            valid_fnr_list,
            valid_recall_list,
            valid_precision_list,
            valid_pos_cosine_sim_list,
            valid_neg_cosine_sim_list,
        ) = calculat_correct_with_threshold(
            all_labels,
            all_imgs,
            all_master_img_names,
            vt_mean,
            vt_std,
            epoch_output_plt_dir,
            all_predicted_cosine_sim,
            cosine_threshold_list,
        )

        tmp_msg += f"Valid pos cosine sim: {plot_distribution(valid_pos_cosine_sim_list, pos_dis_file_name)}\n"
        tmp_msg += f"Valid neg cosine sim: {plot_distribution(valid_neg_cosine_sim_list, neg_dis_file_name)}\n"

        tmp_msg += f"{plot_acc_fp_fn_curves(valid_acc_list, valid_fpr_list, valid_fnr_list, cosine_threshold_list, acc_fp_fn_file_name)}\n"
        tmp_msg += f"{plot_recall_precision_curves(valid_recall_list, valid_precision_list, cosine_threshold_list, recall_precision_file_name)}\n"

        for i in range(len(valid_acc_list)):
            acc = valid_acc_list[i]
            fpr = valid_fpr_list[i]
            fnr = valid_fnr_list[i]
            recall = valid_recall_list[i]
            precision = valid_precision_list[i]
            dis_threshold = cosine_threshold_list[i]

            if i == 0:
                # best_threshold_youden
                valid_acc_list_by_youden.append(acc)
                valid_fp_list_by_youden.append(fpr)
                valid_fn_list_by_youden.append(fnr)
                valid_recall_list_by_youden.append(recall)
                valid_precision_list_by_youden.append(precision)
            else:
                # best_threshold_distance
                valid_acc_list_by_dis.append(acc)
                valid_fp_list_by_dis.append(fpr)
                valid_fn_list_by_dis.append(fnr)
                valid_recall_list_by_dis.append(recall)
                valid_precision_list_by_dis.append(precision)

            tmp_msg += f"Dis threshold: {dis_threshold}, Acc: {acc:.4f}%, Fpr: {fpr:.4f}%, Fnr: {fnr:.4f}%, Recall: {recall:.4f}%, Precision: {precision:.4f}%\n"
            tmp_msg += "---- -----\n\n"

        tmp_msg += f"youden {plot_acc_fp_fn_curves(valid_acc_list_by_youden, valid_fp_list_by_youden, valid_fn_list_by_youden, None, acc_fp_fn_by_youden_file_name)}\n"
        tmp_msg += f"youden {plot_recall_precision_curves(valid_recall_list_by_youden, valid_precision_list_by_youden, None, recall_precision_by_youden_file_name)}\n"

        tmp_msg += f"dis {plot_acc_fp_fn_curves(valid_acc_list_by_dis, valid_fp_list_by_dis, valid_fn_list_by_dis, None, acc_fp_fn_by_dis_file_name)}\n"
        tmp_msg += f"dis {plot_recall_precision_curves(valid_recall_list_by_dis, valid_precision_list_by_dis, None, recall_precision_by_dis_file_name)}\n"

        tmp_msg += "\n=====\n"
        sum_message += tmp_msg
        log.info(f"=== SUM === \n" + sum_message)

        early_stopping(model=model, val_loss=valid_loss, val_acc=-1)
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )
    main()
