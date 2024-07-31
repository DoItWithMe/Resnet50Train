import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from typing import Any
import shutil
from torchvision.transforms import functional as F

from typing import Optional, Callable
from torch import Tensor
import torch.nn.functional as nnF
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision.models.resnet import conv1x1, conv3x3


def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

def l2n(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + "(" + "eps=" + str(self.eps) + ")"


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


class HalfBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class ResNet47_50Net(nn.Module):
    """
    取 resnet50 的 47 层的输出，输出 512 维度
    """

    output_dim = 512

    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        resnet50_model = models.resnet50(pretrained=True)
        features = list(resnet50_model.children())[:-3]

        lay4 = list(resnet50_model.children())[-3]
        lay4[-1] = HalfBottleneck(  # type: ignore
            inplanes=2048,
            planes=512,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=BatchNorm2d,
        )
        features.append(lay4)

        self.features = nn.Sequential(*features)
        self.pool = GeM()
        self.norm = L2N()

        self.lwhiten = None
        self.whiten = None

    def forward(self, x: Tensor):
        # featured_t shape: torch.Size([1, dim, 7, 7])
        featured_t: Tensor = self.features(x)
        pooled_t: Tensor = self.pool(featured_t)
        normed_t: Tensor = self.norm(pooled_t)
        o: Tensor = normed_t.squeeze(-1).squeeze(-1)

        # 使每个图像为Dx1列向量(如果有许多图像，则为DxN)
        return o.permute(0, 1)


class LiftedStructuredLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LiftedStructuredLoss, self).__init__()
        # margin：一个超参数，用于控制损失函数的 margin（边际）。边际用于平衡同类样本和异类样本之间的距离差异。默认为 1.0
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings：表示模型生成的特征嵌入，通常是一个形状为 [n, d] 的张量，其中 n 是样本数量，d 是嵌入的维度。
        # labels：表示样本的标签，形状为 [n] 的张量。
        n = embeddings.size(0)
        # torch.cdist：计算 embeddings 中所有样本对之间的欧氏距离，生成一个形状为 [n, n] 的距离矩阵。
        distances = torch.cdist(embeddings, embeddings)

        # mask：一个布尔矩阵，表示样本对是否具有相同的标签。
        # labels.unsqueeze(0) 和 labels.unsqueeze(1) 分别将 labels 扩展为 [1, n] 和 [n, 1] 的矩阵，通过比较它们来生成一个 [n, n] 的布尔矩阵，表示每对样本是否属于同一类。
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        pos_mask = mask.float()
        neg_mask = (~mask).float()

        # Positive distance
        pos_distances = pos_mask * distances
        pos_distances_sum = torch.sum(pos_distances, dim=1)

        # Negative distance
        neg_distances = neg_mask * distances
        neg_distances_sum = torch.sum(neg_distances, dim=1)

        # Loss calculation
        loss = torch.mean(
            torch.nn.functional.relu(
                self.margin + pos_distances_sum - neg_distances_sum
            )
        )

        return loss


# 自定义数据集，应用数据增强并选择负样本
class ImageNetDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        augmentations=None,
        augmentations_num=10,
        preprocess=None,
        k_neg_samples=-1,
    ):
        self.preprocess = preprocess
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations

        self.dataset: list[str] = list()
        self.dataset_map = dict()
        self.approximate_negs_list: list[list[str]] = list()

        for sub_dir_name in os.listdir(self.dataset_dir):
            sub_dir_path = os.path.join(self.dataset_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                tmp_img_name_path_list = [
                    os.path.join(sub_dir_path, img_name)
                    for img_name in os.listdir(sub_dir_path)
                ]
                tmp_imgs_num = len(tmp_img_name_path_list)

                split_index = int(tmp_imgs_num * 0.7)

                self.dataset.extend(tmp_img_name_path_list[0:split_index])
                self.approximate_negs_list.append(tmp_img_name_path_list[split_index:])

                for tmp_img_path in tmp_img_name_path_list[0:split_index]:
                    self.dataset_map[tmp_img_path] = self.approximate_negs_list[-1]

        self.augmentations_num = augmentations_num
        print(
            f"dateset createm, have dataset: {len(self.dataset)} and approximate_negs_list: {len(self.approximate_negs_list)}"
        )
        self.k_neg_samples = k_neg_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print("call getitem......")
        # print(f"call getitem......index: {index}")

        src_img_path = self.dataset[index]
        # print(f"src_img_path: {src_img_path}")
        src_img = Image.open(src_img_path).convert("RGB")

        src_imgs = list()
        pos_imgs = list()
        src_imgs.append(src_img)

        if self.augmentations:
            for i in range(0, self.augmentations_num):
                src_imgs.append(self.augmentations(src_img))

        if self.preprocess:
            for img in src_imgs:
                pos_imgs.append(self.preprocess(img))

        neg_imgs = list()
        # 从当前母本的特定负样本集合中挑选负样本
        # 15 GB的显存不足以放入所有的负样本
        # print(f"{self.dataset[index]} have {len(self.data_map[src_img_path])} negs........")
        for img in self.dataset_map[src_img_path]:
            # print(f"read Ng: {img}")
            tmp_img = Image.open(img).convert("RGB")
            if self.preprocess:
                tmp_img = self.preprocess(tmp_img)

            neg_imgs.append(tmp_img)
            if len(neg_imgs) >= int(0.5 * self.k_neg_samples):
                break

        # # 将其他母本当成目前母本图片的负样本
        while len(neg_imgs) < self.k_neg_samples:
            random_neg_index = random.randint(0, len(self.dataset) - 1)
            if random_neg_index != index:
                tmp_img = Image.open(self.dataset[random_neg_index]).convert("RGB")
                if self.preprocess:
                    tmp_img = self.preprocess(tmp_img)
                neg_imgs.append(tmp_img)

        pos_labels = [1] * len(pos_imgs)
        neg_labels = [0] * len(neg_imgs)

        imgs = pos_imgs + neg_imgs
        labels = pos_labels + neg_labels

        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.int8),
            self.dataset[index],
        )


# 自定义数据集，应用数据增强并选择负样本
class VtDataSet(Dataset):
    def __init__(
        self,
        dataset_dir,
        augmentations=None,
        augmentations_num=10,
        preprocess=None,
        k_neg_samples=-1,
    ):
        self.preprocess = preprocess
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations

        self.data_map = dict()
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

        if k_neg_samples == -1:
            self.k_neg_samples = int(len(self.dataset) * 0.3)
        else:
            self.k_neg_samples = (
                k_neg_samples
                if k_neg_samples < len(self.dataset)
                else len(self.dataset) - 1
            )
        self.augmentations_num = augmentations_num

        # dump dataset info
        print(
            f"dataset create, have {len(self.dataset)} data, ecache will have at least {self.k_neg_samples} negetive samples..."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_img_path = self.dataset[index]
        # print(f"src_img_path: {src_img_path}")
        src_img = Image.open(src_img_path).convert("RGB")

        src_imgs = list()
        pos_imgs = list()
        src_imgs.append(src_img)

        if self.augmentations:
            for i in range(0, self.augmentations_num):
                src_imgs.append(self.augmentations(src_img))

        if self.preprocess:
            for img in src_imgs:
                pos_imgs.append(self.preprocess(img))

        neg_imgs = list()

        # 从当前母本的特定负样本集合中挑选负样本
        # 15 GB的显存不足以放入所有的负样本
        # print(f"{self.dataset[index]} have {len(self.data_map[src_img_path])} negs........")
        for img in self.data_map[src_img_path]:
            # print(f"read Ng: {img}")
            tmp_img = Image.open(img).convert("RGB")
            if self.preprocess:
                # print(f"process: {img}")
                tmp_img = self.preprocess(tmp_img)

            neg_imgs.append(tmp_img)

            if len(neg_imgs) >= self.k_neg_samples:
                break

        # 将其他母本当成目前母本图片的负样本
        while len(neg_imgs) < self.k_neg_samples:
            random_neg_index = random.randint(0, len(self.dataset) - 1)
            if random_neg_index != index:
                tmp_img = Image.open(self.dataset[random_neg_index]).convert("RGB")
                if self.preprocess:
                    tmp_img = self.preprocess(tmp_img)

                neg_imgs.append(tmp_img)

        pos_labels = [1] * len(pos_imgs)
        neg_labels = [0] * len(neg_imgs)

        imgs = pos_imgs + neg_imgs
        labels = pos_labels + neg_labels

        # if self.is_for_train:
        #     score_labels = [label * 100 for label in labels]
        #     return torch.stack(imgs), torch.tensor(score_labels, dtype=torch.int8)
        # else:
        #     return torch.stack(imgs), torch.tensor(labels, dtype=torch.int8)
        # print(f"data shape: [1,({len(pos_labels)} + {len(neg_labels)})] ")
        # print(
        #     f"data shapessss {torch.stack(imgs).shape}, labels: { torch.tensor(labels, dtype=torch.float32).shape}, pos_imgs: {len(imgs)}"
        # )

        # print(f"imgs: {torch.stack(imgs).shape}, lables: {torch.tensor(labels, dtype=torch.int8).shape}")
        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.int8),
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
def train(model, loader, criterion, optimizer, device, accumulation_steps, max_train_smaples_num):
    model.train()
    running_loss = 0.0

    count = 0
    total_len = len(loader) if max_train_smaples_num == -1 else  max_train_smaples_num
    for i, (samples, labels, _) in enumerate(loader):
        count = i
        # print(f"i: {i}, samples: {samples.shape}, labels: {labels.shape}")

        # Flatten size and labels from [batch_size, num_samples_per_image, channels, height ,width] to [batch_size * num_samples_per_image, channels, height, width]
        flatten_samples, flatten_labels = flatten_data(samples, labels, device)
        # print(
        #     f"i: {i}, samples: {samples.shape}, labels: {labels.shape}, flatten_samples: {flatten_samples.shape}, flatten_labels: {flatten_labels.shape}"
        # )

        # 前向传播（Forward Pass）：计算模型的输出。
        outputs = model(flatten_samples)
        # print(f"i: {i}, outputs: {outputs.shape}, outputs.squeeze(): {outputs.squeeze().shape}")

        # 计算损失（Loss Calculation）：根据模型输出和真实标签计算损失。
        loss = criterion(outputs.squeeze(), flatten_labels)
        loss = loss / accumulation_steps

        # 计算梯度
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # 反向传播（Backward Pass）：计算损失函数相对于模型参数的梯度。
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

        running_loss += loss.item() * flatten_samples.size(0)
        del outputs
        del flatten_samples, flatten_labels

        if count % 100 == 0:
            print(
                f"time:{get_time()}, train {count +1}/{total_len}:{((count + 1) / total_len * 100.0 ):.2f}% in one round..."
            )

        if max_train_smaples_num != -1 and count > max_train_smaples_num:
            break

    if (count + 1) % accumulation_steps != 0:
        # 反向传播（Backward Pass）：计算损失函数相对于模型参数的梯度。
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()

    return running_loss / count


def calculate_l2_distance(tensor1, tensor2, sqrt=True):
    # 由于 vt 系统使用 milvus 使用的是非开根的 l2 距离，为了直观对比结果，这里计算相似度也参考 milvus 的实现
    # 确保张量是浮点型
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # 展平张量
    tensor1_flat = tensor1.view(-1)
    tensor2_flat = tensor2.view(-1)

    # 计算 L2 距离
    difference = tensor1_flat - tensor2_flat
    squared_diff = difference**2
    distance = torch.sum(squared_diff)

    if sqrt:
        return float(torch.sqrt(distance).item())
    else:
        return float(distance.item())


def distance_2_score(distance: float, threshold: float) -> float:
    if threshold > distance:
        return (threshold - distance) / threshold * 100
    else:
        return 0.0


# 验证函数
def validate(model, loader, device, dis_threshold, max_val_smaples_num):
    model.eval()
    # total_running_loss = 0.0

    all_predictions_scores = list()
    all_labels = list()
    all_imgs = list()
    all_dis = list()
    all_master_img_path = list()
    with torch.no_grad():
        for i, (samples, labels, path) in enumerate(loader):
            # 展开 samples 和 labels
            flatten_samples, flatten_labels = flatten_data(samples, labels, device)
            # print(f"flatten_labels: {flatten_labels.shape}")

            # 预测
            outputs = model(flatten_samples)
            predictions_scores = list()
            dis = list()

            for i in range(0, len(flatten_samples)):
                l2_dis = calculate_l2_distance(outputs[0], outputs[i], sqrt=False)
                score = distance_2_score(l2_dis, dis_threshold)
                predictions_scores.append(score)
                dis.append(l2_dis)

                # print(
                #     f"dis: {l2_dis}, score: {score}, dis_threshold: {dis_threshold}, true score: {flatten_labels[i]}"
                # )

                # print(f"predictions_scores: {predictions_scores}")

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_path.append(file_name)
            all_dis.append(dis)
            all_imgs.append(flatten_samples)
            all_predictions_scores.append(predictions_scores)
            all_labels.append(flatten_labels.tolist())
            del outputs
            if max_val_smaples_num != -1 and i > max_val_smaples_num:
                break

    return all_predictions_scores, all_labels, all_imgs, all_dis, all_master_img_path


def merge_images_with_text(
    master_img,
    master_img_denormalize,
    comparison_img,
    comparison_img_denormalize,
    dis,
    score,
    is_correct,
    label,
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
    text = f"Dis: {dis:.2f}, Score: {score:.2f}, is_correct: {is_correct}, label: {bool(label)}"
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


def calculate_correct_with_score_threshold(
    all_predictions_scores,
    all_labels,
    all_imgs,
    all_dis,
    all_master_img_path,
    score_threshold,
    output_plt_dir,
    vt_mean,
    vt_std,
):

    correct = 0.0
    fp_count = 0.0
    fn_count = 0.0
    num = 0.0

    for i in range(0, len(all_imgs)):
        master_img = all_imgs[i][0]
        master_img_denormalize = denormalize(master_img, vt_mean, vt_std)

        num += len(all_imgs[i]) - 1

        for j in range(1, len(all_imgs[i])):
            score = all_predictions_scores[i][j]
            label = all_labels[i][j]
            sample_img = all_imgs[i][j]
            sample_img_denormalize = denormalize(sample_img, vt_mean, vt_std)
            dis = all_dis[i][j]

            is_correct = False
            if label == 1:
                if score >= score_threshold:
                    is_correct = True
                    correct += 1
                else:
                    fn_count += 1
            else:
                if score < score_threshold:
                    is_correct = True
                    correct += 1
                else:
                    fp_count += 1

            output_plt_path = os.path.join(
                output_plt_dir, f"{all_master_img_path[i]}-{j}-label{label}.png"
            )
            # print(f"save img to {output_plt_path}")
            # print("start merge_images_with_text")
            merge_images_with_text(
                master_img,
                master_img_denormalize,
                sample_img,
                sample_img_denormalize,
                dis,
                score,
                is_correct,
                label,
                output_plt_path,
                50,
                100,
            )
            # print("end merge_images_with_text")

    return correct / num * 100.0, fp_count / num * 100.0, fn_count / num * 100.0


k_neg_samples = 40
augmentations_num = 20

vt_mean = [0.485, 0.456, 0.406]
vt_std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose(
    [
        # 调整图像大小
        transforms.Resize([256, 256]),
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
        # 高斯模糊
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomResizedCrop([256, 256]),
        # # 剪裁
        # transforms.RandomResizedCrop(256),
        # transforms.ToTensor(),
    ]
)

train_dataset = ImageNetDataset(
    dataset_dir="/data/jinzijian/assets/ImageNet2012/ILSVRC2012_img_train",
    augmentations=augmentations,
    augmentations_num=augmentations_num,
    preprocess=preprocess,
    k_neg_samples=k_neg_samples,
)

val_k_neg_samples = 5
val_augmentations_num = 5

vt_val_dataset = VtDataSet(
    dataset_dir="/data/jinzijian/resnet50/assets/vt-imgs-for-test",
    augmentations=augmentations,
    augmentations_num=val_augmentations_num,
    preprocess=preprocess,
    k_neg_samples=val_k_neg_samples,
)


num_workers = 10

batch_size = 64

# 梯度累积步数
accumulation_steps = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=int(batch_size / accumulation_steps),
    shuffle=True,
    num_workers=num_workers,
    drop_last=False,
)

vt_val_loader = DataLoader(
    vt_val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Use {device}")

model = ResNet47_50Net()
model = model.to(device)

# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

# 设置损失函数
criterion = LiftedStructuredLoss(margin=1.0)

num_epochs = 30
dis_threshold = 0.6
score_threshold = 80

output_plt_dir = "/data/jinzijian/resnet50/output"
shutil.rmtree(output_plt_dir)
os.mkdir(output_plt_dir)

sum_message = ""

best_model_wts = None
best_acc = 0.0

max_train_smaples_num = 5000
max_val_smaples_num = 50

model_save_dir_path = "/data/jinzijian/resnet50/model-output"
shutil.rmtree(model_save_dir_path)
os.mkdir(model_save_dir_path)
for epoch in range(num_epochs):
    print("start train ---")
    t1 = time.time()
    train_loss = train(
        model, train_loader, criterion, optimizer, device, accumulation_steps, max_train_smaples_num
    )
    t2 = time.time()

    print("start valid -----------")
    all_predictions_scores, all_labels, all_imgs, all_dis, all_master_img_path = (
        validate(model, vt_val_loader, device, dis_threshold, max_val_smaples_num)
    )

    epoch_output_plt_dir = os.path.join(output_plt_dir, str(epoch + 1))
    try:
        shutil.rmtree(epoch_output_plt_dir)
    except Exception as e:
        pass

    os.mkdir(epoch_output_plt_dir)

    val_acc, fp, fn = calculate_correct_with_score_threshold(
        all_predictions_scores,
        all_labels,
        all_imgs,
        all_dis,
        all_master_img_path,
        score_threshold,
        epoch_output_plt_dir,
        vt_mean,
        vt_std,
    )
    del all_predictions_scores, all_labels, all_imgs, all_dis, all_master_img_path

    tmp_msg = f"time: {get_time()}, Device: {device}, Epoch {epoch+1}/{num_epochs}, Train cost: {t2 - t1}s, Train Loss: {train_loss:.4f}\n"
    tmp_msg += f"val_acc: {val_acc:.4f}%, fp: {fp:.4f}%, fn: {fn:.4f}%\n"
    tmp_msg += "=====\n"
    sum_message += tmp_msg
    print(sum_message)
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = model.state_dict().copy()
        model_save_file_path = os.path.join(model_save_dir_path, f"{epoch}.pth")
        torch.save(best_model_wts, model_save_file_path)
        print(f"time: {get_time()}, Best model saved with accuracy: {best_acc:.4f}, train round: {epoch}")
