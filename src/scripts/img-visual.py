import os
from venv import logger

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import shutil
import sys

sys.path.append(_project_dir)
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger as log
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from utils.image_utils import RemoveSolidEdgeRectangle,RandomWatermark,RandomDenoise,Sharpen
from dataset.dataset import VtDataSet
from augly.image import (
    EncodingQuality,
    OneOf,
    RandomBlur,
    RandomEmojiOverlay,
    RandomPixelization,
    RandomRotation,
    ShufflePixels,
)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Create a new tensor for denormalized values
    denormalized_tensor = tensor.clone()

    # Perform denormalization
    for t, m, s in zip(denormalized_tensor, mean, std):
        t.mul_(s).add_(m)

    return denormalized_tensor


def get_all_pos_neg_imgs(dataloader: DataLoader, vt_mean, vt_std):
    ref_list = list()
    pos_list = list()
    neg_list = list()
    total_len = len(dataloader)
    for i, (samples, labels, path) in enumerate(dataloader):
        logger.info(f"visual {i+1}/{total_len}")
        ref = samples[0, 0, :, :, :]
        pos_sample = samples[0, 1, :, :, :]
        neg_sample = samples[0, 2, :, :, :]

        ref_list.append(
            (
                denormalize(ref, vt_mean, vt_std).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(int)
        )
        pos_list.append(
            (
                denormalize(pos_sample, vt_mean, vt_std).permute(1, 2, 0).cpu().numpy()
                * 255
            ).astype(int)
        )
        neg_list.append(
            (
                denormalize(neg_sample, vt_mean, vt_std).permute(1, 2, 0).cpu().numpy()
                * 255
            ).astype(int)
        )
        # return ref_list, pos_list, neg_list

    return ref_list, pos_list, neg_list


def merge_images_with_text(
    master_img_denormalize_np,
    pos_img_denormalize_np,
    neg_img_denormalize_np,
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
    # 创建左右两边的白色间隔
    left_right_padding = (
        np.ones((master_img_denormalize_np.shape[0], side_padding, 3), dtype=np.uint8)
        * 0
    )  # 白色间隔

    # 创建上下白色间隔
    top_bottom_padding_master = (
        np.ones(
            (bottom_padding, master_img_denormalize_np.shape[1] + 2 * side_padding, 3),
            dtype=np.uint8,
        )
        * 0
    )  # 白色间隔

    top_bottom_padding_comparison = (
        np.ones(
            (
                bottom_padding,
                neg_img_denormalize_np.shape[1] + 2 * side_padding,
                3,
            ),
            dtype=np.uint8,
        )
        * 0
    )  # 白色间隔

    # 为每张图像添加左右白色间隔
    master_img_denormalize_padded = np.concatenate(
        (left_right_padding, master_img_denormalize_np, left_right_padding), axis=1
    )

    neg_img_denormalize_padded = np.concatenate(
        (left_right_padding, neg_img_denormalize_np, left_right_padding), axis=1
    )

    pos_img_denormalize_padded = np.concatenate(
        (left_right_padding, pos_img_denormalize_np, left_right_padding), axis=1
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

    neg_img_denormalize_padded = np.concatenate(
        (
            top_bottom_padding_comparison,
            neg_img_denormalize_padded,
            top_bottom_padding_comparison,
        ),
        axis=0,
    )

    pos_img_denormalize_padded = np.concatenate(
        (
            top_bottom_padding_comparison,
            pos_img_denormalize_padded,
            top_bottom_padding_comparison,
        ),
        axis=0,
    )

    # 确定间隔的宽度和高度
    gap_array = (
        np.ones((master_img_denormalize_padded.shape[0], gap, 3), dtype=np.uint8) * 0
    )  # 白色间隔

    # log.info(
    #     f" master_img_denormalize_padded.shape[1]: { master_img_denormalize_padded.shape[1]}"
    # )
    # log.info(
    #     f" pos_img_denormalize_padded.shape[1]: { pos_img_denormalize_padded.shape[1]}"
    # )
    # log.info(
    #     f" neg_img_denormalize_padded.shape[1]: { neg_img_denormalize_padded.shape[1]}"
    # )
    
    # print(
    # type(title_height),
    # type(master_img_denormalize_padded.shape[1]),
    # type(gap),
    # type(pos_img_denormalize_padded[1]),
    # type(neg_img_denormalize_padded.shape[1]),
    # )

    title_gap_array = (
        np.ones(
            (
                title_height,
                master_img_denormalize_padded.shape[1]
                + gap
                + pos_img_denormalize_padded.shape[1]
                + gap
                + neg_img_denormalize_padded.shape[1]
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
            pos_img_denormalize_padded,
            gap_array,
            neg_img_denormalize_padded,
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


def draw_all_imgs(ref_list, pos_list, neg_list, output_plt_dir):

    for idx, ref in enumerate(ref_list):
        output_plt_path = os.path.join(output_plt_dir, f"{idx}.png")
        pos = pos_list[idx]
        neg = neg_list[idx]

        merge_images_with_text(
            ref,
            pos,
            neg,
            0,
            -999,
            1,
            False,
            output_plt_path,
            50,
            100,
        )
        log.info(f"save to {output_plt_path}")


def main():
    vt_mean = [0.485, 0.456, 0.406]
    vt_std = [0.229, 0.224, 0.225]

    watermark_text = ["English", "中文.English", "中文水印", "asdas中文"]

    meihao_preprocess = transforms.Compose(
        [
            RemoveSolidEdgeRectangle(tolerance=0),
            # 调整图像大小
            transforms.Resize(256),
            # 中心剪裁
            transforms.CenterCrop(224),
            # 随机灰度值, 训练时可以这样，为了能方便的让正负样本都随机变成灰度图, 上线使用不需要
            # transforms.RandomGrayscale(p=0.5),
            RandomWatermark(watermark_text, font_size=20, opacity_range=(50, 150)),
            # 转换为张量，并将像素值缩放到[0, 1]
            transforms.ToTensor(),
            # NormalizePerImage(),
            transforms.Normalize(mean=vt_mean, std=vt_std),
        ]
    )

    # augmentations = transforms.Compose(
    #     [
    #         RemoveSolidEdgeRectangle(tolerance=0),
    #         # 水平翻转
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         # 旋转
    #         transforms.RandomRotation(degrees=15),
    #         # 先 resize
    #         transforms.Resize(256),
    #         # 随机剪裁
    #         transforms.RandomResizedCrop([224, 224], scale=(0.5, 1.0)),
    #         # 高斯模糊
    #         transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    #         # 亮度变化, 对比度，饱和度，色调
    #         transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=[0.0, 0.1]),  # type: ignore
    #         RandomWatermark(watermark_text, font_size=20, opacity_range=(50, 150)),
    #         # h264 h265 常见锐化，降噪操作
    #         # 降噪
    #         RandomDenoise(h_range=(1, 10)),
    #         # 锐化
    #         Sharpen(),
    #         # 转换为张量，并将像素值缩放到[0, 1]
    #         transforms.ToTensor(),
    #         # NormalizePerImage(),
    #         transforms.Normalize(mean=vt_mean, std=vt_std),
    #     ]
    # )
    
    augmentations = transforms.Compose(
        [
            RemoveSolidEdgeRectangle(tolerance=0),
            # 水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 旋转
            transforms.RandomRotation(degrees=15),
            # 先 resize
            transforms.Resize(256),
            # 随机剪裁
            transforms.RandomResizedCrop([224, 224], scale=(0.5, 1.0)),
            # 随机像素化再恢复，有助于模型处理模糊细节或者细节较少的图像
            RandomPixelization(p=0.25),
            # 随机透视变换，模拟不同视角的变化，提高模型的空间不变性
            transforms.RandomPerspective(p=0.25),
            # 随机使用一个编码质量标准对图像进行重新编码，模拟图像质量下降的情况，替代之前的高斯模糊
            OneOf([EncodingQuality(quality=q) for q in [10, 20, 30, 50]], p=0.25),
            # 亮度变化, 对比度，饱和度，色调
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=[0.0, 0.1]),  # type: ignore
            transforms.RandomGrayscale(p=0.3),
            RandomWatermark(watermark_text, font_size=20, opacity_range=(50, 150)),
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
    assets_base_dir_path = "/data/jinzijian/resnet50/assets"

    meihao_train_dataset = VtDataSet(
        dataset_dir="/data/jinzijian/resnet50/assets/vt-imgs-train",
        augmentations=augmentations,
        preprocess=meihao_preprocess,
    )

    batch_size = 1
    num_workers = 2

    meihao_train_loader = DataLoader(
        meihao_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    ref_list, pos_list, neg_list = get_all_pos_neg_imgs(
        meihao_train_loader, vt_mean, vt_std
    )

    # ref_list = ref_list[:2]
    # pos_list = pos_list[:2]
    # neg_list = neg_list[:2]

    output_plt_dir = "/data/jinzijian/resnet50/img-visual-output"
    shutil.rmtree(output_plt_dir)
    os.makedirs(output_plt_dir, exist_ok=True)
    draw_all_imgs(ref_list, pos_list, neg_list, output_plt_dir)


if __name__ == "__main__":
    main()
