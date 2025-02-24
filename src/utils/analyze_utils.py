import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from collections import Counter
from sklearn.metrics import roc_curve, auc
import seaborn as sns


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
    best_threshold = thresholds[best_threshold_index]
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
        label=f"Best threshold by J: {best_threshold:.2f}",
    )
    plt.scatter(
        fpr[best_threshold_index_distance],
        tpr[best_threshold_index_distance],
        marker="x",
        color="g",
        label=f"Best threshold by distance: {best_threshold_distance:.2f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(filename)
    plt.close()

    msg = f"roc_auc: {roc_auc}, Best threshold by Youdens J:{best_threshold}, Best threshold by distance: {best_threshold_distance}, saved to {filename}"
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
    plt.plot(train_losses, label="Training Loss", color="blue")

    # 绘制验证损失曲线
    plt.plot(val_losses, label="Validation Loss", color="red")

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


def plot_acc_fp_fn_curves(acc_list, fp_list, fn_list, l2_dis_threshold_list, filename):
    """
    绘制训练损失和验证损失的变化曲线。

    参数:
    - train_losses (list of float): 训练损失列表
    - val_losses (list of float): 验证损失列表
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    plt.plot(l2_dis_threshold_list, acc_list, label="Acc", color="blue", marker="x")

    plt.plot(l2_dis_threshold_list, fp_list, label="FP", color="red", marker="o")

    plt.plot(l2_dis_threshold_list, fn_list, label="FN", color="green", marker="v")

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Acc && FP && FN")
    plt.xlabel("L2 Threshold")
    plt.ylabel("Percentage")
    plt.xticks(
        ticks=l2_dis_threshold_list,
        labels=[f"{x:.2f}" for x in l2_dis_threshold_list],
        rotation=45,
    )

    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


def plot_recall_precision_curves(
    recall_list, precision_list, l2_dis_threshold_list, filename
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

    plt.plot(
        l2_dis_threshold_list, recall_list, label="recall", color="blue", marker="x"
    )

    plt.plot(
        l2_dis_threshold_list,
        precision_list,
        label="precision",
        color="red",
        marker="o",
    )

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title("Recall && Precison")
    plt.xlabel("L2 Threshold")
    plt.ylabel("Percentage")
    plt.xticks(
        ticks=l2_dis_threshold_list,
        labels=[f"{x:.2f}" for x in l2_dis_threshold_list],
        rotation=45,
    )

    plt.savefig(filename)
    plt.close()

    msg = f"saved to {filename}"
    return msg


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
                output_plt_dir, f"{all_master_img_names[i]}-{j}-label{label}.png"
            )
            # merge_images_with_text(
            #     master_img,
            #     master_img_denormalize,
            #     sample_img,
            #     sample_img_denormalize,
            #     dis,
            #     -999,
            #     label,
            #     False,
            #     output_plt_path,
            #     50,
            #     100,
            # )
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
