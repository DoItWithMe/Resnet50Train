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

    def extractor_features(self, x):
        results_map = dict()
        fc1_x = self.forward_one(x)
        fc1_x = nnF.normalize(fc1_x)
        results_map["fc1"] = fc1_x

        pool_x = self.feature_extractor(x)
        pool_x = torch.flatten(pool_x, 1)
        pool_x = nnF.normalize(pool_x)
        results_map["resnet50_avgpool_out"] = pool_x

        lay4_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        lay4 = lay4_feature_extractor(x)
        lay4: Tensor = torch.flatten(lay4, 1)
        lay4: Tensor = nnF.normalize(lay4)
        results_map["resnet50_lay4_output"] = lay4

        return results_map

    def extractor_features_by_fc1(self, x):
        return self.forward_one(x)

    def extractor_features_by_resnet(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x


class SiameseNetwork3(nn.Module):
    def __init__(self, resetnet50_weight_path):
        super(SiameseNetwork3, self).__init__()
        # 使用预训练的 ResNet50 作为特征提取器
        self.resnet = models.resnet50(weights=None)
        pretrained_weight = torch.load(resetnet50_weight_path)
        self.resnet.load_state_dict(pretrained_weight["state_dict"], strict=False)

        # 移除 ResNet50 的全连接层
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

    def extractor_features(self, x):
        results_map = dict()

        pool_x = self.feature_extractor(x)
        pool_x = torch.flatten(pool_x, 1)
        pool_x = nnF.normalize(pool_x)
        results_map["resnet50_avgpool_out"] = pool_x

        lay4_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        lay4 = lay4_feature_extractor(x)
        lay4: Tensor = torch.flatten(lay4, 1)
        lay4: Tensor = nnF.normalize(lay4)
        results_map["resnet50_lay4_output"] = lay4

        return results_map

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

    def extractor_features(self, x):
        results_map = dict()

        pool_x = self.feature_extractor(x)
        pool_x = torch.flatten(pool_x, 1)
        pool_x = nnF.normalize(pool_x)
        results_map["resnet50_avgpool_out"] = pool_x

        lay4_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        lay4 = lay4_feature_extractor(x)
        lay4: Tensor = torch.flatten(lay4, 1)
        lay4: Tensor = nnF.normalize(lay4)
        results_map["resnet50_lay4_output"] = lay4

        return results_map

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


# 验证函数
def validate(model, loader, device, criterion, max_val_smaples_num):
    model.eval()
    running_loss = 0.0
    valid_count = 0.0

    all_predicted_probabilities = list()
    all_labels = list()
    all_imgs = list()
    all_master_img_names = list()
    total_len = len(loader) if max_val_smaples_num < 0 else max_val_smaples_num

    all_l2_dis = dict()
    all_cosine_sim = dict()

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

            outputs = model(valid_refs, valid_samples)
            loss = criterion(outputs, valid_labels)
            running_loss += loss.item()

            valid_refs_feats = model.extractor_features(valid_refs)
            valid_samples_feats = model.extractor_features(valid_samples)

            for k, ref_feats in valid_refs_feats.items():
                if k not in all_l2_dis.keys():
                    all_l2_dis[k] = list()
                if k not in all_cosine_sim.keys():
                    all_cosine_sim[k] = list()

                sample_feats = valid_samples_feats[k]

                l2_dis = nnF.pairwise_distance(ref_feats, sample_feats)
                cos_dis = nnF.cosine_similarity(ref_feats, sample_feats, dim=1) + 1
                all_l2_dis[k].append([l2_dis.to("cpu").tolist()[0]])
                all_cosine_sim[k].append([cos_dis.to("cpu").tolist()[0]])

            valid_count += 1
            if int(valid_count) % 100 == 0:
                log.info(
                    f"valid {valid_count}/{total_len}:{((valid_count ) / (total_len) * 100.0 ):.2f}% in one round...\n"
                )

            # 通过 sigmoid 将模型输出映射到 [0, 1] 区间
            probabilities = torch.sigmoid(outputs.squeeze())

            del valid_samples
            del valid_labels
            del outputs
            torch.cuda.empty_cache()

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_names.append(file_name)
            flatten_samples, flatten_labels = flatten_data(samples, labels, "cpu")
            all_labels.append(flatten_labels.to("cpu").tolist())
            all_imgs.append(flatten_samples.to("cpu"))
            all_predicted_probabilities.append([probabilities.to("cpu").tolist()])

    return (
        running_loss / valid_count,
        all_predicted_probabilities,
        all_labels,
        all_imgs,
        all_master_img_names,
        all_l2_dis,
        all_cosine_sim,
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
    return msg, best_threshold_by_youden, best_threshold_by_distance


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


def calculat_correct_with_threshold(
    all_labels,
    all_imgs,
    all_master_img_names,
    vt_mean,
    vt_std,
    output_plt_dir,
    all_predicted_probabilities,
    threshold_list,
):  # -> tuple[list[float], list[float], list[float], list[float],...:

    num = 0.0
    tp_list = [0 for i in range(len(threshold_list))]
    tn_list = [0 for i in range(len(threshold_list))]
    fp_list = [0 for i in range(len(threshold_list))]
    fn_list = [0 for i in range(len(threshold_list))]

    fpr_list = [0.0 for i in range(len(threshold_list))]
    fnr_list = [0.0 for i in range(len(threshold_list))]
    acc_list = [0.0 for i in range(len(threshold_list))]
    recall_list = [0.0 for i in range(len(threshold_list))]
    precision_list = [0.0 for i in range(len(threshold_list))]

    neg_probabilities_list = list()
    pos_probabilities_list = list()

    for i in range(0, len(all_imgs)):
        master_img = all_imgs[i][0]
        master_img_denormalize = denormalize(master_img, vt_mean, vt_std)
        num += len(all_imgs[i]) - 1

        for j in range(0, len(all_labels[i])):
            sample_img = all_imgs[i][j + 1]
            sample_img_denormalize = denormalize(sample_img, vt_mean, vt_std)

            probabilities = all_predicted_probabilities[i][j]
            label = all_labels[i][j]

            output_plt_path = os.path.join(
                output_plt_dir, f"{all_master_img_names[i]}-{j}-label{int(label)}.png"
            )
            merge_images_with_text(
                master_img,
                master_img_denormalize,
                sample_img,
                sample_img_denormalize,
                probabilities,
                -999,
                label,
                False,
                output_plt_path,
                50,
                100,
            )

            if label == 1:
                pos_probabilities_list.append(probabilities)
            else:
                neg_probabilities_list.append(probabilities)

            for k, threshold in enumerate(threshold_list):
                # log.info(f"lable: {label} -- {label == 1}")
                predicted_label = 1 if probabilities >= threshold else 0
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
        pos_probabilities_list,
        neg_probabilities_list,
    )


def calculat_correct_with_l2_dis_threshold(
    all_labels,
    all_predicted_dis,
    l2_dis_threshold_list,
):

    num = 0.0

    tp_list = [0 for i in range(len(l2_dis_threshold_list))]
    tn_list = [0 for i in range(len(l2_dis_threshold_list))]
    fp_list = [0 for i in range(len(l2_dis_threshold_list))]
    fn_list = [0 for i in range(len(l2_dis_threshold_list))]

    fpr_list = [0.0 for i in range(len(l2_dis_threshold_list))]
    fnr_list = [0.0 for i in range(len(l2_dis_threshold_list))]
    acc_list = [0.0 for i in range(len(l2_dis_threshold_list))]
    recall_list = [0.0 for i in range(len(l2_dis_threshold_list))]
    precision_list = [0.0 for i in range(len(l2_dis_threshold_list))]

    neg_dis_list = list()
    pos_dis_list = list()

    for i in range(0, len(all_labels)):
        num += len(all_labels[i])

        for j in range(0, len(all_labels[i])):
            dis = all_predicted_dis[i][j]
            label = all_labels[i][j]
            if label == 1:
                pos_dis_list.append(dis)
            else:
                neg_dis_list.append(dis)

            for k, l2_dis_threshold in enumerate(l2_dis_threshold_list):
                predicted_label = 1 if dis < l2_dis_threshold else 0
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
        pos_dis_list,
        neg_dis_list,
    )


def calculat_correct_with_cosine_sim_threshold(
    all_labels,
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

    for i in range(0, len(all_labels)):
        num += len(all_labels[i])

        for j in range(0, len(all_labels[i])):
            cosine_sim = all_predicted_cosine_sim[i][j]
            label = all_labels[i][j]
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


def plot_distribution(data, filename, need_box=True):
    if len(data) == 0:
        return "None"
    if need_box:
        mean = np.mean(data)
        median = np.median(data)
        mode = Counter(data).most_common(1)[0][0]
        variance = np.var(data)
        std_dev = np.std(data)

        msg = f"avg: {mean:.2f}, median: {median:.2f}, mode: {mode:.2f}, variance: {variance:.2f}, std_dev: {std_dev:.2f}, saved to {filename}"
    else:
        msg = f"saved to {filename}"

    # 3. 绘制分布图
    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=100, kde=True)
    plt.title("Histogram")

    if need_box:
        # 盒须图
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data)
        plt.title("Box and whisker plot")

    # 保存图像
    plt.savefig(filename)
    plt.close()
    return msg


def plot_acc_fp_fn_histogram(acc_list, fp_list, fn_list, threshold_list, filename):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    bar_width = 0.25  # 每个柱的宽度
    index = np.arange(len(acc_list))  # X轴的位置

    if threshold_list is not None:
        plt.bar(index, acc_list, bar_width, label="Acc", color="blue")
        plt.bar(index + bar_width, fp_list, bar_width, label="FP", color="red")
        plt.bar(index + 2 * bar_width, fn_list, bar_width, label="FN", color="green")

        plt.xticks(
            ticks=index + bar_width,
            labels=[f"{x:.2f}" for x in threshold_list],
            rotation=45,
        )
        plt.xlabel("Threshold")
    else:
        plt.bar(index, acc_list, bar_width, label="Acc", color="blue")
        plt.bar(index + bar_width, fp_list, bar_width, label="FP", color="red")
        plt.bar(index + 2 * bar_width, fn_list, bar_width, label="FN", color="green")

        plt.xticks(
            ticks=index + bar_width,
            labels=[f"Round {i+1}" for i in range(len(acc_list))],
        )
        plt.xlabel("Round")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Acc && FP && FN")
    plt.ylabel("Percentage")

    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def plot_recall_precision_histogram(
    recall_list, precision_list, cosine_threshold_list, filename
):
    """
    绘制召回率(Recall)和精确率(Precision)的直方图, 并在 X 轴上显示对应的阈值或轮次。

    参数:
    - recall_list (list of float): 召回率列表
    - precision_list (list of float): 精确率列表
    - cosine_threshold_list (list of float): 阈值列表，对应 X 轴刻度
    - filename (str): 图像保存路径
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    bar_width = 0.35  # 每个柱的宽度
    index = np.arange(len(recall_list))  # X轴的位置

    if cosine_threshold_list is not None:
        plt.bar(index, recall_list, bar_width, label="Recall", color="blue")
        plt.bar(
            index + bar_width, precision_list, bar_width, label="Precision", color="red"
        )

        plt.xticks(
            ticks=index + bar_width / 2,
            labels=[f"{x:.2f}" for x in cosine_threshold_list],
            rotation=45,
        )
        plt.xlabel("Threshold")
    else:
        plt.bar(index, recall_list, bar_width, label="Recall", color="blue")
        plt.bar(
            index + bar_width, precision_list, bar_width, label="Precision", color="red"
        )

        plt.xticks(
            ticks=index + bar_width / 2,
            labels=[f"Round {i+1}" for i in range(len(recall_list))],
        )
        plt.xlabel("Round")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Recall & Precision Histogram")
    plt.ylabel("Percentage")

    # 保存图像
    plt.savefig(filename)
    plt.close()

    msg = f"Histogram saved to {filename}"
    return msg


def anaylize(
    valid_loss,
    all_predicted_probabilities,
    all_labels,
    all_imgs,
    all_master_img_names,
    all_l2_dis,
    all_cosine_sim,
    device,
    model,
    model_weight_file_path,
    valid_round,
    data_distribution_plot_save_dir_path,
    vt_mean,
    vt_std,
    output_plt_dir,
):
    auc_file_name = os.path.join(data_distribution_plot_save_dir_path, "auc_plot.png")

    pos_probabilities_file_name = os.path.join(
        data_distribution_plot_save_dir_path, "pos_probabilities_plot.png"
    )

    neg_probabilities_file_name = os.path.join(
        data_distribution_plot_save_dir_path, "neg_probabilities_plot.png"
    )

    acc_fp_fn_file_name = os.path.join(
        data_distribution_plot_save_dir_path, "acc_fpr_fnr_plot.png"
    )

    recall_precision_file_name = os.path.join(
        data_distribution_plot_save_dir_path, "recall_presion_plot.png"
    )

    tmp_msg = f"Device: {device}, use model: {type(model)}, weight file path: {model_weight_file_path}, valid_loss: {valid_loss/valid_round}\n"

    auc_msg, threshold_by_youden, threshold_by_dis = plot_roc_auc(
        all_labels, all_predicted_probabilities, auc_file_name
    )

    tmp_msg += f"Valid auc: {auc_msg}\n"
    threshold_list = [0.5, threshold_by_youden, threshold_by_dis]
    # 基于预测结果计算 acc,fp,fn,recall,precision
    (
        valid_acc_list,
        valid_fpr_list,
        valid_fnr_list,
        valid_recall_list,
        valid_precision_list,
        valid_pos_probabilities_list,
        valid_neg_probabilities_list,
    ) = calculat_correct_with_threshold(
        all_labels,
        all_imgs,
        all_master_img_names,
        vt_mean,
        vt_std,
        output_plt_dir,
        all_predicted_probabilities,
        threshold_list,
    )

    tmp_msg += f"Valid pos probabilities: {plot_distribution(valid_pos_probabilities_list, pos_probabilities_file_name)}\n"
    tmp_msg += f"Valid neg probabilities: {plot_distribution(valid_neg_probabilities_list, neg_probabilities_file_name)}\n"
    tmp_msg += f"Valid Acc/Fp/Fn: {plot_acc_fp_fn_histogram(valid_acc_list ,valid_fpr_list, valid_fnr_list, threshold_list, acc_fp_fn_file_name)}"
    tmp_msg += f"Valid Recall/Precision {plot_recall_precision_histogram(valid_recall_list, valid_precision_list, threshold_list, recall_precision_file_name)}\n"

    for i in range(len(valid_acc_list)):
        acc = valid_acc_list[i]
        fpr = valid_fpr_list[i]
        fnr = valid_fnr_list[i]
        recall = valid_recall_list[i]
        precision = valid_precision_list[i]
        threshold = threshold_list[i]

        tmp_msg += f"Threshold: {threshold}, Acc: {acc:.4f}%, Fpr: {fpr:.4f}%, Fnr: {fnr:.4f}%, Recall: {recall:.4f}%, Precision: {precision:.4f}%\n"
        tmp_msg += "\n\n"
    # 绘制距离分布图
    tmp_msg += "Analyze data based on l2 distance\n"
    l2_dis_threshold_list = [float(i) for i in np.arange(0.0, 5.0, 0.5)]
    for k in all_l2_dis.keys():
        tmp_predicted_dis = all_l2_dis[k]

        tmp_acc_fp_fn_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"acc_fpr_fnr_plot_l2_dis_{k}.png"
        )

        tmp_recall_precision_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"recall_presion_plot_ls_dis_{k}.png"
        )

        tmp_pos_dis_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"pos_plot_l2_dis_{k}.png"
        )

        tmp_neg_dis_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"neg_plot_l2_dis_{k}.png"
        )

        (
            tmp_acc_list,
            tmp_fpr_list,
            tmp_fnr_list,
            tmp_recall_list,
            tmp_precision_list,
            tmp_pos_dis_list,
            tmp_neg_dis_list,
        ) = calculat_correct_with_l2_dis_threshold(
            all_labels,
            tmp_predicted_dis,
            l2_dis_threshold_list,
        )

        tmp_msg += f"featrue type: {k}, Valid pos l2 dis: {plot_distribution(tmp_pos_dis_list, tmp_pos_dis_file_name)}\n"
        tmp_msg += f"featrue type: {k}, Valid neg l2 dis: {plot_distribution(tmp_neg_dis_list, tmp_neg_dis_file_name)}\n"
        tmp_msg += f"featrue type: {k}, {plot_acc_fp_fn_histogram(tmp_acc_list, tmp_fpr_list, tmp_fnr_list, l2_dis_threshold_list, tmp_acc_fp_fn_file_name)}\n"
        tmp_msg += f"featrue type: {k}, {plot_recall_precision_histogram(tmp_recall_list, tmp_precision_list, l2_dis_threshold_list, tmp_recall_precision_file_name)}\n"
        tmp_msg += "\n\n"

    tmp_msg += "Analyze data based on cosine similarly\n"
    for k in all_cosine_sim.keys():
        tmp_predicted_cosine_sim = all_cosine_sim[k]
        tmp_auc_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"auc_plot_cosine_sim_{k}.png"
        )

        tmp_acc_fp_fn_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"acc_fpr_fnr_plot_cosine_sim_{k}.png"
        )

        tmp_recall_precision_file_name = os.path.join(
            data_distribution_plot_save_dir_path,
            f"recall_presion_plot_cosine_sim_{k}.png",
        )

        tmp_pos_cos_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"pos_plot_cosine_sim_{k}.png"
        )

        tmp_neg_cos_file_name = os.path.join(
            data_distribution_plot_save_dir_path, f"neg_plot_cosine_sim_{k}.png"
        )

        tmsg, best_threshold_by_youden, best_threshold_by_distance = plot_roc_auc(
            all_labels, tmp_predicted_cosine_sim, tmp_auc_file_name
        )
        tmp_msg += f"featrue type: {k}, {tmsg}\n"
        cosine_threshold_list = [best_threshold_by_youden, best_threshold_by_distance]
        (
            tmp_acc_list,
            tmp_fpr_list,
            tmp_fnr_list,
            tmp_recall_list,
            tmp_precision_list,
            tmp_pos_cosine_sim_list,
            tmp_neg_cosine_sim_list,
        ) = calculat_correct_with_cosine_sim_threshold(
            all_labels,
            tmp_predicted_cosine_sim,
            cosine_threshold_list,
        )

        tmp_msg += f"featrue type: {k}, Valid pos cosine sim: {plot_distribution(tmp_pos_cosine_sim_list, tmp_pos_cos_file_name)}\n"
        tmp_msg += f"featrue type: {k}, Valid neg cosine sim: {plot_distribution(tmp_neg_cosine_sim_list, tmp_neg_cos_file_name)}\n"
        tmp_msg += f"featrue type: {k}, {plot_acc_fp_fn_histogram(tmp_acc_list, tmp_fpr_list, tmp_fnr_list, cosine_threshold_list, tmp_acc_fp_fn_file_name)}\n"
        tmp_msg += f"featrue type: {k}, {plot_recall_precision_histogram(tmp_recall_list, tmp_precision_list, cosine_threshold_list, tmp_recall_precision_file_name)}\n"
        tmp_msg += "\n\n"

    tmp_msg += "\n=====\n"
    log.info(f"=== SUM === \n" + tmp_msg)


def main(
    assets_base_dir_path,
    model_weight_file_path,
    model,
    device,
    max_val_smaples_num,
    valid_round,
):
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

    log.info(f"Use {device}\n")

    gpu_count = torch.cuda.device_count()

    vt_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "vt-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    vt_val_loader = DataLoader(
        vt_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    if gpu_count <= 1:
        model = model.to(device)
    else:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    sum_message = ""

    output_plt_dir = os.path.join(output_dir, "imgs")
    try:
        shutil.rmtree(output_plt_dir)
    except Exception as e:
        pass

    os.makedirs(output_plt_dir, exist_ok=True)

    data_distribution_plot_save_dir_path = os.path.join(output_dir, "data_distribution")
    try:
        shutil.rmtree(data_distribution_plot_save_dir_path)
    except Exception as e:
        pass

    os.makedirs(data_distribution_plot_save_dir_path, exist_ok=True)

    all_predicted_probabilities = list()
    all_labels = list()
    all_imgs = list()
    all_master_img_names = list()
    valid_loss = 0.0

    all_l2_dis = dict()
    all_cosine_sim = dict()

    for round in range(valid_round):
        log.info(f"start valid, round: {round+1}/{valid_round}...\n")
        (
            tmp_valid_loss,
            tmp_all_predicted_probabilities,
            tmp_all_labels,
            tmp_all_imgs,
            tmp_all_master_img_names,
            tmp_all_l2_dis,
            tmp_all_cosine_sim,
        ) = validate(model, vt_val_loader, device, criterion, max_val_smaples_num)
        all_predicted_probabilities.extend(tmp_all_predicted_probabilities)
        all_labels.extend(tmp_all_labels)

        all_imgs = tmp_all_imgs
        all_master_img_names = tmp_all_master_img_names

        for k, v in tmp_all_l2_dis.items():
            if k not in all_l2_dis.keys():
                all_l2_dis[k] = list()
            if k not in all_cosine_sim.keys():
                all_cosine_sim[k] = list()

            all_l2_dis[k].extend(v)
            all_cosine_sim[k].extend(tmp_all_cosine_sim[k])

        valid_loss += tmp_valid_loss

    anaylize(
        valid_loss,
        all_predicted_probabilities,
        all_labels,
        all_imgs,
        all_master_img_names,
        all_l2_dis,
        all_cosine_sim,
        device,
        model,
        model_weight_file_path,
        valid_round,
        data_distribution_plot_save_dir_path,
        vt_mean,
        vt_std,
        output_plt_dir,
    )


if __name__ == "__main__":
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )

    assets_base_dir_path = "/data/jinzijian/resnet50/assets"
    device = torch.device("cpu")
    max_val_smaples_num = -1
    valid_round = 5

    # Valid Test 9
    # model = SiameseNetwork2(
    #     resetnet50_weight_path=os.path.join(
    #         assets_base_dir_path, "based-models", "gl18-tl-resnet50-gem-w-83fdc30.pth"
    #     )
    # )
    # model_weight_file_path = os.path.join(
    #     assets_base_dir_path, "best-models", "test_9_2024-09-04-09_23_06.pth"
    # )
    # model_weight = torch.load(model_weight_file_path)
    # model.load_state_dict(model_weight)
    # model_weight_file_name = os.path.basename(model_weight_file_path).split(".pth")[0]
    # output_dir = os.path.join(
    #     "/data/jinzijian/resnet50/valid-output/", model_weight_file_name
    # )

    # Valid Test 13
    # model = SiameseNetwork3(
    #     resetnet50_weight_path=os.path.join(
    #         assets_base_dir_path, "based-models", "gl18-tl-resnet50-gem-w-83fdc30.pth"
    #     )
    # )
    # model_weight_file_path = os.path.join(
    #     assets_base_dir_path, "best-models", "test_13_2024-09-10-03_33_51.pth"
    # )
    # model_weight = torch.load(model_weight_file_path)
    # model.load_state_dict(model_weight)
    # model_weight_file_name = os.path.basename(model_weight_file_path).split(".pth")[0]
    # output_dir = os.path.join(
    #     "/data/jinzijian/resnet50/valid-output/", model_weight_file_name
    # )
    
    # Valid Test 14
    model = SiameseNetwork4()
    model_weight_file_path = os.path.join(
        assets_base_dir_path, "best-models", "test_14_2024-09-11-02_06_48.pth"
    )
    model_weight = torch.load(model_weight_file_path)
    model.load_state_dict(model_weight)
    model_weight_file_name = os.path.basename(model_weight_file_path).split(".pth")[0]
    output_dir = os.path.join(
        "/data/jinzijian/resnet50/valid-output/", model_weight_file_name
    )

    main(
        assets_base_dir_path,
        model_weight_file_path,
        model,
        device,
        max_val_smaples_num,
        valid_round,
    )
