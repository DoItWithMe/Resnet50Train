from torch.utils.data import Dataset
import os
import multiprocessing
from loguru import logger as log
from PIL import Image
import random
import torch

# 自定义数据集，应用数据增强并选择负样本
class VtDataSet(Dataset):
    def __init__(
        self,
        dataset_dir,
        augmentations,
        preprocess,
        another_tyep_neg_probability=0.5,
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
                dataset_dir, os.path.splitext(data_path)[0], "neg"
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

        self.another_tyep_neg_probability = another_tyep_neg_probability

        log.info(f"dataset_dir: {dataset_dir} --- {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_img_path = self.dataset[index]

        src_img = Image.open(src_img_path).convert("RGB")

        imgs = list()
        labels = list()
        process_src_img = self.preprocess(src_img)
        imgs.append(process_src_img)

        imgs.append(self.augmentations(src_img))
        labels.append(1)

        rand_number = random.random()
        if rand_number <= self.another_tyep_neg_probability:
            neg_total_num = len(self.data_map[src_img_path])
            neg_index = random.randint(0, neg_total_num - 1)

            img = self.data_map[src_img_path][neg_index]
            neg_img = Image.open(img).convert("RGB")
            imgs.append(self.preprocess(neg_img))
            labels.append(0)
        else:
            rand_type_index = random.randint(0, len(self.dataset) - 1)
            neg_total_num = len(self.data_map[self.dataset[rand_type_index]]) - 1
            rand_neg_index = random.randint(0, neg_total_num)

            img = self.data_map[self.dataset[rand_type_index]][rand_neg_index]
            neg_img = Image.open(img).convert("RGB")
            imgs.append(self.preprocess(neg_img))
            labels.append(0)

        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.float),
            self.dataset[index],
        )


# 自定义数据集，应用数据增强并选择负样本
class ImageNetDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        augmentations,
        preprocess,
        another_tyep_neg_probability=0.5,
        downsample_rate=0.01,
    ):
        self.preprocess = preprocess
        self.dataset_dir = dataset_dir
        self.augmentations = augmentations
        self.another_tyep_neg_probability = another_tyep_neg_probability
        self.downsample_rate = downsample_rate
        self.dataset: list[str] = list()
        self.data_map = dict()

        self.rebuild()

    def rebuild(self):
        self.dataset.clear()
        self.data_map.clear()

        approximate_negs_list: list[list[str]] = list()

        for sub_dir_name in os.listdir(self.dataset_dir):
            sub_dir_path = os.path.join(self.dataset_dir, sub_dir_name)
            if os.path.isdir(sub_dir_path):
                tmp_img_name_path_list = [
                    os.path.join(sub_dir_path, img_name)
                    for img_name in os.listdir(sub_dir_path)
                ]
                random.shuffle(tmp_img_name_path_list)
                tmp_imgs_num = (
                    len(tmp_img_name_path_list) * self.downsample_rate
                )  # from 0.05 -> 0.01
                # log.info(f"tmp_img_name_path_list: {len(tmp_img_name_path_list)}, tmp_imgs_num: {tmp_imgs_num}")

                split_index = int(tmp_imgs_num * 0.5)

                self.dataset.extend(tmp_img_name_path_list[0:split_index])
                approximate_negs_list.append(tmp_img_name_path_list[split_index:])

                for tmp_img_path in tmp_img_name_path_list[0:split_index]:
                    self.data_map[tmp_img_path] = approximate_negs_list[-1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_img_path = self.dataset[index]

        src_img = Image.open(src_img_path).convert("RGB")

        imgs = list()
        labels = list()
        process_src_img = self.preprocess(src_img)
        imgs.append(process_src_img)

        imgs.append(self.augmentations(src_img))
        labels.append(1)

        rand_number = random.random()
        if rand_number <= self.another_tyep_neg_probability:
            neg_total_num = len(self.data_map[src_img_path])
            neg_index = random.randint(0, neg_total_num - 1)

            img = self.data_map[src_img_path][neg_index]
            neg_img = Image.open(img).convert("RGB")
            imgs.append(self.preprocess(neg_img))
            labels.append(0)
        else:
            rand_type_index = random.randint(0, len(self.dataset) - 1)
            neg_total_num = len(self.data_map[self.dataset[rand_type_index]]) - 1
            rand_neg_index = random.randint(0, neg_total_num)

            img = self.data_map[self.dataset[rand_type_index]][rand_neg_index]
            neg_img = Image.open(img).convert("RGB")
            imgs.append(self.preprocess(neg_img))
            labels.append(0)

        return (
            torch.stack(imgs),
            torch.tensor(labels, dtype=torch.float),
            self.dataset[index],
        )

