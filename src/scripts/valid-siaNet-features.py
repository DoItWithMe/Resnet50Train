import os

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(_project_dir)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as nnF

import torchvision.transforms as transforms

from PIL import Image, ImageFilter
import numpy as np

import matplotlib.pyplot as plt
from typing import Any
import shutil
import cv2


import sys
from loguru import logger as log
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.metrics import roc_curve, auc

from dataset.dataset import VtDataSet
from model.siamese_net import SiameseNetworkL2Net1


def flatten_normalize(x):
    return nnF.normalize(torch.flatten(x, 1))


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, labels):
        # 计算欧式距离
        norm_output1 = nnF.normalize(output1)
        norm_output2 = nnF.normalize(output2)

        euclidean_distance = nnF.pairwise_distance(
            norm_output1, norm_output2, keepdim=True
        )

        # 计算损失
        loss = torch.mean(
            labels * torch.pow(euclidean_distance, 2)
            + (1 - labels)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

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


# 验证函数
def validate(model, loader, device, criterion, max_val_smaples_num, criterion_2):
    model.eval()
    running_loss = 0.0
    valid_count = 0.0

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
            valid_pos_samples = samples[:, 1, :, :, :].to(device)
            valid_neg_samples = samples[:, 2, :, :, :].to(device)
            valid_labels = labels.to(device)

            valid_refs_feats = model.extractor_features(valid_refs)
            valid_pos_samples_feats = model.extractor_features(valid_pos_samples)
            valid_neg_samples_feats = model.extractor_features(valid_neg_samples)

            for k, ref_feats in valid_refs_feats.items():
                if k not in all_l2_dis.keys():
                    all_l2_dis[k] = list()
                if k not in all_cosine_sim.keys():
                    all_cosine_sim[k] = list()

                pos_sample_feats = valid_pos_samples_feats[k]
                neg_sample_feats = valid_neg_samples_feats[k]

                pos_l2_dis = nnF.pairwise_distance(ref_feats, pos_sample_feats)
                neg_l2_dis = nnF.pairwise_distance(ref_feats, neg_sample_feats)

                pos_cos_dis = (
                    nnF.cosine_similarity(ref_feats, pos_sample_feats, dim=1) + 1
                )
                neg_cos_dis = (
                    nnF.cosine_similarity(ref_feats, neg_sample_feats, dim=1) + 1
                )

                all_l2_dis[k].append(
                    [
                        pos_l2_dis.to("cpu").tolist()[0] ** 2,
                        neg_l2_dis.to("cpu").tolist()[0] ** 2,
                    ]
                )
                all_cosine_sim[k].append(
                    [
                        pos_cos_dis.to("cpu").tolist()[0],
                        neg_cos_dis.to("cpu").tolist()[0],
                    ]
                )

            valid_count += 1
            if int(valid_count) % 100 == 0:
                log.info(
                    f"valid {valid_count}/{total_len}:{((valid_count ) / (total_len) * 100.0 ):.2f}% in one round...\n"
                )

            del valid_pos_samples, valid_neg_samples
            del valid_labels
            torch.cuda.empty_cache()

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_names.append(file_name)
            flatten_samples, flatten_labels = flatten_data(samples, labels, "cpu")
            all_labels.append(flatten_labels.to("cpu").tolist())
            all_imgs.append(flatten_samples.to("cpu"))

    return (
        running_loss / valid_count,
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
    # text = f"label: {bool(label)}, predicted label: {bool(predicted_label)}, predicted output: {predicted_output:.5f}, is_correct: {is_correct}, "
    dis_thresh = 0.35
    score = (dis_thresh - predicted_output) / dis_thresh * 100.0
    text = f"predicted score: {score:.2f} "

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
    all_imgs,
    all_master_img_names,
    vt_mean,
    vt_std,
    output_plt_dir,
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

    for i in range(0, len(all_imgs)):
        master_img = all_imgs[i][0]
        master_img_denormalize = denormalize(master_img, vt_mean, vt_std)
        num += len(all_imgs[i]) - 1

        for j in range(0, len(all_labels[i])):
            sample_img = all_imgs[i][j + 1]
            sample_img_denormalize = denormalize(sample_img, vt_mean, vt_std)

            dis = all_predicted_dis[i][j]
            label = all_labels[i][j]

            output_plt_path = os.path.join(
                output_plt_dir,
                f"{all_master_img_names[i]}-{i}-label{int(label)}.png",
            )

            merge_images_with_text(
                master_img,
                master_img_denormalize,
                sample_img,
                sample_img_denormalize,
                dis,
                -999,
                label,
                False,
                output_plt_path,
                50,
                100,
            )

            if label == 1:
                pos_dis_list.append(dis)
            else:
                neg_dis_list.append(dis)

            for k, l2_dis_threshold in enumerate(l2_dis_threshold_list):
                predicted_label = 1 if dis <= l2_dis_threshold else 0
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


def image2(data, filename):
    """
    绘制模型性能对比图
    :param data:
    :return:
    """
    # import matplotlib.pyplot as plt
    # import numpy as np

    # 提取数据
    model_names = [item[0] for item in data]
    metrics = np.array([item[1:] for item in data])

    # 设置图表
    labels = [
        "Accuracy",
        "False Positives",
        "False Negatives",
        "Recall",
        "Precision",
        "F1 Score",
    ]
    num_metrics = len(labels)
    x = np.arange(num_metrics)  # 性能指标的索引
    width = 0.15  # 小柱子的宽度

    base_width = 13
    additional_width = 2 * len(model_names)
    fig, ax = plt.subplots(figsize=(base_width + additional_width, 10))

    # 绘制每个性能指标的小柱子
    for i in range(len(model_names)):
        bars = ax.bar(x + i * width, metrics[i], width, label=model_names[i])
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.2f}",
                ha="center",
            )

    # 添加标签和标题
    ax.set_ylabel("Percentage")
    ax.set_title("Resnet0925 95 score")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Models")

    ax.set_ylim(0, 150)
    ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")

    # 显示图表
    plt.tight_layout()

    # 保存图片到文件，宽度自适应
    plt.savefig(filename)
    plt.close()
    msg = f"Histogram saved to {filename}"
    return msg


def anaylize(
    valid_loss,
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
    val_dataset_name,
):

    tmp_msg = f"Device: {device}, use model: {type(model)}, weight file path: {model_weight_file_path}, valid_loss: {valid_loss/valid_round}\n"

    # 绘制距离分布图
    tmp_msg += "Analyze data based on l2 distance\n"
    l2_dis_threshold_list = [float(i) for i in np.arange(0.0, 3.0, 0.1)]
    # l2_dis_threshold_list = [float(i) for i in np.arange(0.0, 0.5, 0.05)]

    l2_dis_threshold_list = [0.35]
    # log.info(f"all_l2_dis: {all_l2_dis}")
    for k in all_l2_dis.keys():
        tmp_predicted_dis = all_l2_dis[k]
        # log.info(f"tmp_predicted_dis == {k} -- {tmp_predicted_dis}")
        tmp_acc_fp_fn_file_name = os.path.join(
            data_distribution_plot_save_dir_path,
            f"{val_dataset_name}_acc_fpr_fnr_plot_l2_dis_{k}.png",
        )

        tmp_recall_precision_file_name = os.path.join(
            data_distribution_plot_save_dir_path,
            f"{val_dataset_name}_recall_presion_plot_ls_dis_{k}.png",
        )

        tmp_pos_dis_file_name = os.path.join(
            data_distribution_plot_save_dir_path,
            f"{val_dataset_name}_pos_plot_l2_dis_{k}.png",
        )

        tmp_neg_dis_file_name = os.path.join(
            data_distribution_plot_save_dir_path,
            f"{val_dataset_name}_neg_plot_l2_dis_{k}.png",
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
            all_imgs,
            all_master_img_names,
            vt_mean,
            vt_std,
            output_plt_dir,
            tmp_predicted_dis,
            l2_dis_threshold_list,
        )

        tmp_msg += f"featrue type: {k}, Valid pos l2 dis: {plot_distribution(tmp_pos_dis_list, tmp_pos_dis_file_name)}\n"
        tmp_msg += f"featrue type: {k}, Valid neg l2 dis: {plot_distribution(tmp_neg_dis_list, tmp_neg_dis_file_name)}\n"
        tmp_msg += f"featrue type: {k}, {plot_acc_fp_fn_histogram(tmp_acc_list, tmp_fpr_list, tmp_fnr_list, l2_dis_threshold_list, tmp_acc_fp_fn_file_name)}\n"
        # tmp_msg += f"featrue type: {k}, {plot_recall_precision_histogram(tmp_recall_list, tmp_precision_list, l2_dis_threshold_list, tmp_recall_precision_file_name)}\n"

        result = []
        result.append(
            [
                "resnet50",
                tmp_acc_list[0],
                tmp_fpr_list[0],
                tmp_fnr_list[0],
                tmp_recall_list[0],
                tmp_precision_list[0],
                2
                * (tmp_precision_list[0] * tmp_recall_list[0])
                / (tmp_precision_list[0] + tmp_recall_list[0]),
            ]
        )
        tmp_msg += f"tmp img, featrue type: {k}, {image2(result, tmp_recall_precision_file_name)}\n"

        for i in range(len(l2_dis_threshold_list)):
            tmp_msg += f"l2 dis thresh: {l2_dis_threshold_list[i]:.2f}, Recall: {tmp_recall_list[i]:.2f}%, Precision: {tmp_precision_list[i]:.2f}%\n"
        tmp_msg += "\n\n"

    # tmp_msg += "Analyze data based on cosine similarly\n"

    # tmp_labels = list()
    # for i in range(len(all_labels)):
    #     for j in range(len(all_labels[i])):
    #         tmp_labels.append(all_labels[i][j])

    # for k in all_cosine_sim.keys():
    #     tmp_auc_file_name = os.path.join(
    #         data_distribution_plot_save_dir_path, f"auc_plot_cosine_sim_{k}.png"
    #     )

    #     tmp_acc_fp_fn_file_name = os.path.join(
    #         data_distribution_plot_save_dir_path, f"acc_fpr_fnr_plot_cosine_sim_{k}.png"
    #     )

    #     tmp_recall_precision_file_name = os.path.join(
    #         data_distribution_plot_save_dir_path,
    #         f"recall_presion_plot_cosine_sim_{k}.png",
    #     )

    #     tmp_pos_cos_file_name = os.path.join(
    #         data_distribution_plot_save_dir_path, f"pos_plot_cosine_sim_{k}.png"
    #     )

    #     tmp_neg_cos_file_name = os.path.join(
    #         data_distribution_plot_save_dir_path, f"neg_plot_cosine_sim_{k}.png"
    #     )

    #     tmp_predicted_cosine_sim = list()
    #     for i in range(len(all_cosine_sim[k])):
    #         for j in range(len(all_cosine_sim[k][i])):
    #             tmp_predicted_cosine_sim.append(all_cosine_sim[k][i][j])

    #     tmsg, best_threshold_by_youden, best_threshold_by_distance = plot_roc_auc(
    #         tmp_labels, tmp_predicted_cosine_sim, tmp_auc_file_name
    #     )

    #     tmp_msg += f"featrue type: {k}, {tmsg}\n"
    #     cosine_threshold_list = [best_threshold_by_youden, best_threshold_by_distance]
    #     (
    #         tmp_acc_list,
    #         tmp_fpr_list,
    #         tmp_fnr_list,
    #         tmp_recall_list,
    #         tmp_precision_list,
    #         tmp_pos_cosine_sim_list,
    #         tmp_neg_cosine_sim_list,
    #     ) = calculat_correct_with_cosine_sim_threshold(
    #         all_labels,
    #         all_cosine_sim[k],
    #         cosine_threshold_list,
    #     )

    #     tmp_msg += f"featrue type: {k}, Valid pos cosine sim: {plot_distribution(tmp_pos_cosine_sim_list, tmp_pos_cos_file_name)}\n"
    #     tmp_msg += f"featrue type: {k}, Valid neg cosine sim: {plot_distribution(tmp_neg_cosine_sim_list, tmp_neg_cos_file_name)}\n"
    #     tmp_msg += f"featrue type: {k}, {plot_acc_fp_fn_histogram(tmp_acc_list, tmp_fpr_list, tmp_fnr_list, cosine_threshold_list, tmp_acc_fp_fn_file_name)}\n"
    #     tmp_msg += f"featrue type: {k}, {plot_recall_precision_histogram(tmp_recall_list, tmp_precision_list, cosine_threshold_list, tmp_recall_precision_file_name)}\n"
    #     tmp_msg += "\n\n"

    tmp_msg += "\n=====\n"
    log.info(f"=== SUM === \n" + tmp_msg)


class RemoveSolidEdgeRectangle:
    def __init__(self, tolerance: int = 0):
        self.tolerance = tolerance

    def __call__(self, img) -> Any:
        img = img.convert("RGB")
        data = np.array(img)
        height, width, _ = data.shape

        top, bottom, left, right = 0, height - 1, 0, width - 1

        if height > width:
            # 只检查上下边缘
            while top < height:
                if np.all(data[top, :, :] == data[top, 0, :]):  # 检查整行是否一致
                    top += 1
                else:
                    break

            while bottom >= 0:
                if np.all(data[bottom, :, :] == data[bottom, 0, :]):  # 检查整行是否一致
                    bottom -= 1
                else:
                    break

            # 确保裁剪后的高度 >= 宽度
            if (bottom - top + 1) < width:
                center = (top + bottom) // 2
                top = max(0, center - width // 2)
                bottom = min(height - 1, center + width // 2)

        else:
            # 只检查左右边缘
            while left < width:
                if np.all(data[:, left, :] == data[0, left, :]):  # 检查整列是否一致
                    left += 1
                else:
                    break

            while right >= 0:
                if np.all(data[:, right, :] == data[0, right, :]):  # 检查整列是否一致
                    right -= 1
                else:
                    break

            # 确保裁剪后的宽度 >= 高度
            if (right - left + 1) < height:
                center = (left + right) // 2
                left = max(0, center - height // 2)
                right = min(width - 1, center + height // 2)

        new_data = data[top : bottom + 1, left : right + 1, :]
        new_image = Image.fromarray(new_data, "RGB")
        return new_image


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
            RemoveSolidEdgeRectangle(),
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
            RemoveSolidEdgeRectangle(),
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

    meihao_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "meihaojinxiang-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    night_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "night-scenery-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    vt_val_loader = DataLoader(
        vt_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    meihao_val_loader = DataLoader(
        meihao_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    night_val_loader = DataLoader(
        night_val_dataset, batch_size=1, shuffle=False, num_workers=3, drop_last=False
    )

    vt_dataloader_name = "vt"
    meihao_dataloader_name = "meihao"
    night_scenery_dataloader_name = "night_scenery"

    val_dataloader_dict = {
        vt_dataloader_name: vt_val_loader,
        # meihao_dataloader_name: meihao_val_loader,
        # night_scenery_dataloader_name: night_val_loader,
    }

    model = model.to(device)

    # if gpu_count <= 1:
    #     model = model.to(device)
    # else:
    #     model = nn.DataParallel(model)

    criterion = nn.TripletMarginLoss(2.0, 2)
    criterion_2 = ContrastiveLoss(2.0)

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

    for val_name, val_loader in val_dataloader_dict.items():
        log.info(f"start valid dataset: {val_name}")
        all_labels = list()
        all_imgs = list()
        all_master_img_names = list()
        valid_loss = 0.0

        all_l2_dis = dict()
        all_cosine_sim = dict()

        for round in range(valid_round):
            log.info(f"round: {round+1}/{valid_round}...\n")
            (
                tmp_valid_loss,
                tmp_all_labels,
                tmp_all_imgs,
                tmp_all_master_img_names,
                tmp_all_l2_dis,
                tmp_all_cosine_sim,
            ) = validate(
                model, val_loader, device, criterion, max_val_smaples_num, criterion_2
            )
            all_labels.extend(tmp_all_labels)
            # log.info(f"len: {len(tmp_all_imgs)} -- {len(tmp_all_imgs[0])}")
            all_imgs.extend(tmp_all_imgs)
            all_master_img_names.extend(tmp_all_master_img_names)
            # all_imgs = tmp_all_imgs
            # all_master_img_names = tmp_all_master_img_names

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
            val_name,
        )


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module.") :]  # 去掉 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


if __name__ == "__main__":
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )

    assets_base_dir_path = "/data/jinzijian/resnet50/assets"
    device = torch.device("cpu")
    max_val_smaples_num = -100
    valid_round = 1

    # Valid Test 15
    model = SiameseNetworkL2Net1(
        resetnet50_weight_path=os.path.join(
            assets_base_dir_path, "based-models", "gl18-tl-resnet50-gem-w-83fdc30.pth"
        )
    )
    # model_weight_file_path = os.path.join(
    #     assets_base_dir_path, "best-models", "test_15_2024-09-12-10_08_17.pth"
    # )

    # model_weight_file_path = os.path.join(
    #     assets_base_dir_path, "best-models", "test_17_2024-09-25-10_35_16.pth"
    # )

    model_weight_file_path = os.path.join(
        assets_base_dir_path,
        "best-models",
        "test_18_2024-10-12-10_24_49.pth",
    )

    model_weight_file_path = os.path.join(
        assets_base_dir_path,
        "best-models",
        "reset-net-src.pth",
    )

    # logger.info("")
    # print("fake sleep")
    # time.sleep(100)
    # exit(0)

    # model_weight_file_path = "/data/jinzijian/resnet50/output/2024-12-12-09_49_40/model/2024-12-12-09_49_50_115.pth"

    # model_weight_file_path = os.path.join(
    #     assets_base_dir_path, "best-models", "test_20_2024-11-18-02_38_58.pth"
    # )

    # model = SiameseNetworkL2Net2()
    # model_weight_file_path = "/data/jinzijian/resnet50/output/2024-12-13-02_13_13/model/2024-12-13-02_13_22_449.pth"

    # model_weight = torch.load(model_weight_file_path, weights_only=True)
    # model_weight = remove_module_prefix(model_weight)
    # model.load_state_dict(model_weight)
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
