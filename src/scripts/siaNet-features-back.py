import os

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.append(_project_dir)

import shutil
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import numpy as np
import time
from loguru import logger as log

from utils.image_utils import (
    RemoveSolidEdgeRectangle,
    RandomWatermark,
    RandomDenoise,
    Sharpen,
)
from utils.analyze_utils import (
    calculat_correct_with_l2_dis_threshold,
    plot_acc_fp_fn_curves,
    plot_distribution,
    plot_loss_curves,
    plot_recall_precision_curves,
)
from utils.normal_utils import get_time_for_name_some
from utils.model_utils import EarlyStopping, train, validate
from loss.loss import CustomTripletMarginLossWithSwap
from dataset.dataset import VtDataSet, ImageNetDataset
from model.siamese_net import SiameseNetworkL2Net1WithSE


def main():

    vt_mean = [0.485, 0.456, 0.406]
    vt_std = [0.229, 0.224, 0.225]

    watermark_text = ["English", "中文.English", "中文水印", "asdas中文"]

    preprocess = transforms.Compose(
        [
            RemoveSolidEdgeRectangle(tolerance=0),
            # 调整图像大小
            transforms.Resize(256),
            # 中心剪裁
            transforms.CenterCrop(224),
            # 随机灰度值, 训练时可以这样，为了能方便的让正负样本都随机变成灰度图, 上线使用不需要
            transforms.RandomGrayscale(p=0.3),
            RandomWatermark(watermark_text, font_size=20, opacity_range=(50, 150)),
            # 转换为张量，并将像素值缩放到[0, 1]
            transforms.ToTensor(),
            # NormalizePerImage(),
            transforms.Normalize(mean=vt_mean, std=vt_std),
        ]
    )

    meihao_preprocess = transforms.Compose(
        [
            RemoveSolidEdgeRectangle(tolerance=0),
            # 调整图像大小
            transforms.Resize(256),
            # 中心剪裁
            transforms.CenterCrop(224),
            # 随机灰度值, 训练时可以这样，为了能方便的让正负样本都随机变成灰度图, 上线使用不需要
            transforms.RandomGrayscale(p=0.3),
            RandomWatermark(watermark_text, font_size=20, opacity_range=(50, 150)),
            # 转换为张量，并将像素值缩放到[0, 1]
            transforms.ToTensor(),
            # NormalizePerImage(),
            transforms.Normalize(mean=vt_mean, std=vt_std),
        ]
    )

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
            # 高斯模糊
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    gpu_count = torch.cuda.device_count()
    gpu_count = 1

    log.info(f"Use {device} --- gpu_count: {gpu_count}\n")

    assets_base_dir_path = "/data/jinzijian/resnet50/assets"

    output_dir = f"/data/jinzijian/resnet50/output/{get_time_for_name_some()}"

    vt_train_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "vt-imgs-train"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    meihao_train_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "meihaojinxiang-imgs-train"),
        augmentations=augmentations,
        preprocess=meihao_preprocess,
    )

    imgnet_train_dataset = ImageNetDataset(
        dataset_dir="/data/jinzijian/assets/ILSVRC2012_img_train",
        augmentations=augmentations,
        preprocess=preprocess,
    )

    night_train_dataset = VtDataSet(
        dataset_dir="/data/jinzijian/resnet50/assets/night-scenery-imgs-train",
        augmentations=augmentations,
        preprocess=preprocess,
    )

    vt_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "vt-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    meihao_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "meihaojinxiang-imgs-valid"),
        augmentations=augmentations,
        preprocess=meihao_preprocess,
    )

    night_val_dataset = VtDataSet(
        dataset_dir=os.path.join(assets_base_dir_path, "night-scenery-imgs-valid"),
        augmentations=augmentations,
        preprocess=preprocess,
    )

    # return

    num_workers = 6

    total_batch_size = 256

    batch_size = 32 * gpu_count

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

    meihao_train_loader = DataLoader(
        meihao_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    imgnet_train_loader = DataLoader(
        imgnet_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    night_train_loader = DataLoader(
        night_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
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
    imgnet_dataloader_name = "imgnet"
    night_scenery_dataloader_name = "night_scenery"

    train_dataloader_dict = {
        vt_dataloader_name: vt_train_loader,
        meihao_dataloader_name: meihao_train_loader,
        imgnet_dataloader_name: imgnet_train_loader,
        night_scenery_dataloader_name: night_train_loader,
    }

    val_dataloader_dict = {
        vt_dataloader_name: vt_val_loader,
        meihao_dataloader_name: meihao_val_loader,
        night_scenery_dataloader_name: night_val_loader,
    }

    # 创建基于 ResNet50 的孪生网络

    model = SiameseNetworkL2Net1WithSE(
        resetnet50_weight_path="/data/jinzijian/resnet50/assets/based-models/seresnet50_ra_224-8efdb4bb.pth"
    )

    # model = SiameseNetworkL2Net3()

    # 将模型迁移至多个 GPU 设备如果可行
    if gpu_count <= 1:
        model = model.to(device)
    else:
        model = nn.DataParallel(model)
        model = model.to(device)

    # 设置优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    # init_lr = batch_size / 256
    optimizer = optim.SGD(  # type:ignore
        model.parameters(), 0.03, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=0
    )
    # optim.lr_scheduler.CosineAnnealingWarmRestarts()
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)

    # 设置学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

    # 设置损失函数
    criterion = CustomTripletMarginLossWithSwap(margin=4.0, p=2, swap=True)
    # criterion = nn.TripletMarginLoss(margin=3.5, p=2, swap=False)
    # scaler = GradScaler()

    sum_message = ""

    num_epochs = 3000
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
        patience=1000,
        delta=0.01,
        path=model_save_file_path,
        mode="min_loss",
        trace_func=log.info,
    )

    # vt_train_loss_list = list()
    # imgnet_train_loss_list = list()
    valid_loss_dict: dict[str, list[float | Any]] = dict()

    train_loss_dict: dict[str, list[float | Any]] = dict()

    l2_dis_threshold_list = [float(i) for i in np.arange(0.0, 5, 0.5)]

    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, init_lr, epoch, num_epochs)
        t1 = time.time()

        imgnet_train_dataset.rebuild()
        train_dataloader_dict[imgnet_dataloader_name] = DataLoader(
            imgnet_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

        for name, train_dataloader in train_dataloader_dict.items():
            log.info(f"start train {name}")
            train_loss = train(
                model,
                train_dataloader,
                criterion,
                optimizer,
                None,
                device,
                accumulation_steps,
                max_train_smaples_num,
            )

            if name not in train_loss_dict.keys():
                train_loss_dict[name] = list()
            train_loss_dict[name].append(train_loss)

        # imgnet_dataset_train_loss = 1.0

        t2 = time.time()
        scheduler.step()
        log.info(f"lr: {scheduler.get_last_lr()[0]}")

        epoch_output_plt_dir = os.path.join(
            output_plt_dir,
            f"{str(epoch + 1)}",
        )
        try:
            shutil.rmtree(epoch_output_plt_dir)
        except Exception as e:
            pass

        os.mkdir(epoch_output_plt_dir)

        epoch_data_distribution_plot_dir = os.path.join(
            data_distribution_plot_save_dir_path, str(epoch + 1)
        )
        try:
            shutil.rmtree(epoch_data_distribution_plot_dir)
        except Exception as e:
            pass

        os.mkdir(epoch_data_distribution_plot_dir)
        avg_valid_loss = 0.0
        for val_name, val_dataloader in val_dataloader_dict.items():
            log.info(f"start valid {val_name}\n")
            (
                valid_loss,
                all_labels,
                all_imgs,
                all_master_img_names,
                all_predicted_dis,
            ) = validate(model, val_dataloader, device, criterion, max_val_smaples_num)

            if val_name not in valid_loss_dict:
                valid_loss_dict[val_name] = list()
            valid_loss_dict[val_name].append(valid_loss)
            avg_valid_loss = avg_valid_loss + valid_loss

            log.info("start calculate acc/fp/fn...")
            (
                valid_acc_list,
                valid_fpr_list,
                valid_fnr_list,
                valid_recall_list,
                valid_precision_list,
                valid_pos_dis_list,
                valid_neg_dis_list,
            ) = calculat_correct_with_l2_dis_threshold(
                all_labels,
                all_imgs,
                all_master_img_names,
                vt_mean,
                vt_std,
                epoch_output_plt_dir,
                all_predicted_dis,
                l2_dis_threshold_list,
            )

            loss_curves_file_name_vt = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_vt_loss_curves.png"
            )

            loss_curves_file_name_imgnet = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_imgnet_loss_curves.png"
            )

            pos_dis_file_name = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_pos_dis_plot.png"
            )

            neg_dis_file_name = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_neg_dis_plot.png"
            )

            acc_fp_fn_file_name = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_acc_fpr_fnr_plot.png"
            )

            recall_precision_file_name = os.path.join(
                epoch_data_distribution_plot_dir, f"{val_name}_recall_presion_plot.png"
            )

            tmp_msg = f"Device: {device}, Epoch {epoch+1}/{num_epochs}, "
            tmp_msg += f"Train cost: {(t2 - t1):.3f}s, "
            for loss_name, loss_list in train_loss_dict.items():
                tmp_msg += f"{loss_name}_loss: {loss_list[-1]:.4f}, "

            tmp_msg += f"\n"
            for loss_name, loss_list in train_loss_dict.items():
                if loss_name == imgnet_dataloader_name:
                    continue
                tmp_msg += f"loss curves: {plot_loss_curves(loss_list, valid_loss_dict[val_name], os.path.join(epoch_data_distribution_plot_dir, f'{loss_name}_loss_curves.png'))}\n"

            tmp_msg += f"Valid pos dis: {plot_distribution(valid_pos_dis_list, pos_dis_file_name)}\n"
            tmp_msg += f"Valid neg dis: {plot_distribution(valid_neg_dis_list, neg_dis_file_name)}\n"
            tmp_msg += f"{plot_acc_fp_fn_curves(valid_acc_list, valid_fpr_list, valid_fnr_list, l2_dis_threshold_list, acc_fp_fn_file_name)}\n"
            tmp_msg += f"{plot_recall_precision_curves(valid_recall_list, valid_precision_list, l2_dis_threshold_list, recall_precision_file_name)}\n"

            for i in range(len(valid_acc_list)):
                acc = valid_acc_list[i]
                fpr = valid_fpr_list[i]
                fnr = valid_fnr_list[i]
                recall = valid_recall_list[i]
                precision = valid_precision_list[i]
                dis_threshold = l2_dis_threshold_list[i]

                tmp_msg += f"Dis threshold: {dis_threshold:.2f}, Acc: {acc:.4f}%, Fpr: {fpr:.4f}%, Fnr: {fnr:.4f}%, Recall: {recall:.4f}%, Precision: {precision:.4f}%\n"
                tmp_msg += "---- -----\n\n"

            tmp_msg += "\n=====\n"
            log.info(f"=== SUM === \n" + tmp_msg)

        avg_valid_loss = avg_valid_loss / len(val_dataloader_dict.keys())
        early_stopping(
            model=model, val_loss=avg_valid_loss, val_acc=-1, epoch=epoch + 1
        )
        if early_stopping.early_stop:
            break


if __name__ == "__main__":
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )
    main()
