import os
import torch
from loguru import logger as log
# import torch.nn as nn

# class GeneralizedMeanPooling(nn.Module):
#     def __init__(self, p=3):
#         super(GeneralizedMeanPooling, self).__init__()
#         self.p = p

#     def forward(self, x):
#         # x 的形状为 [batch_size, channels, height, width]
#         batch_size, channels, height, width = x.size()

#         # 将特征图展平
#         x_flat = x.view(
#             batch_size, channels, -1
#         )  # 形状为 [batch_size, channels, height * width]

#         # 计算均值和最大值
#         x_mean = x_flat.mean(dim=2)  # [batch_size, channels]
#         x_max = x_flat.max(dim=2)[0]  # [batch_size, channels]

#         # 应用 Generalized Mean Pooling
#         x_pooled = (x_mean**self.p + x_max**self.p) / 2
#         x_pooled = x_pooled ** (1 / self.p)  # [batch_size, channels]

#         return x_pooled


# def gem(x, p=3, eps=1e-6):
#     return nnF.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
#         1.0 / p
#     )


def flatten_data(samples, labels, device):
    batch_size = samples.size(0)
    num_samples_per_image = samples.size(1)
    channels = samples.size(2)
    height = samples.size(3)
    width = samples.size(4)

    return samples.view(-1, channels, height, width).to(device), labels.view(-1).to(
        device
    )


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

    def __call__(self, val_loss, val_acc, model, epoch):
        if self.mode == "min_loss":
            score = -val_loss
        else:  # mode == 'max_acc'
            score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model, epoch)
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
            self.save_checkpoint(val_loss, val_acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model, epoch):
        file_name, file_ext = os.path.splitext(self.path)

        save_path = f"{file_name}_{epoch}{file_ext}"
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

        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            # 如果是，保存 model.module 的参数
            torch.save(model.module.state_dict(), save_path)
        else:
            # 如果不是，直接保存模型的参数
            torch.save(model.state_dict(), save_path)


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * (1 - (epoch / epochs))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


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

    # accumulation_steps = 8

    total_len = len(loader) if max_train_smaples_num < 0 else max_train_smaples_num

    for i, (samples, labels, _) in enumerate(loader):
        if max_train_smaples_num > 0 and train_count >= max_train_smaples_num:
            log.info(f"train: {train_count} / {max_train_smaples_num}, stop train...\n")
            break

        train_refs = samples[:, 0, :, :, :].to(device)
        train_pos_samples = samples[:, 1, :, :, :].to(device)
        train_neg_samples = samples[:, 2, :, :, :].to(device)
        train_labels = labels.to(device)

        o1, o2, o3 = model(train_refs, train_pos_samples, train_neg_samples)
        # log.info(f"outputs: {outputs} ---- train_labels: {train_labels.shape} -- {train_labels}")
        # 计算损失
        tripe_loss = criterion((o1), (o2), (o3))
        # contrastive_loss_pos = criterion_2(
        #     o1, o2, torch.ones(samples.shape[0]).to(device)
        # )
        # loss = 0.6 * tripe_loss + 0.4 * contrastive_loss_pos
        loss = tripe_loss

        del train_refs, train_pos_samples, train_neg_samples, train_labels, o1, o2, o3
        torch.cuda.empty_cache()

        loss = loss / accumulation_steps

        # 反向传播和优化
        loss.backward()

        if i % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()

        train_count += 1.0

        batzh_size = samples.shape[0]
        if int(train_count) % 5 == 0:
            log.info(
                f"train {train_count * batzh_size}/{total_len * batzh_size}:{((train_count ) / (total_len  ) * 100.0 ):.2f}% in one round...\n"
            )

    if i % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / train_count * accumulation_steps


# 验证函数
def validate(model, loader, device, criterion, max_val_smaples_num):
    model.eval()
    running_loss = 0.0
    valid_count = 0.0

    all_predicted_dis = list()
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
            valid_pos_samples = samples[:, 1, :, :, :].to(device)
            valid_neg_samples = samples[:, 2, :, :, :].to(device)

            valid_labels = labels.to(device)

            o1, o2, o3 = model(valid_refs, valid_pos_samples, valid_neg_samples)
            pos_dis = torch.nn.functional.pairwise_distance(o1, o2) ** 2
            neg_dis = torch.nn.functional.pairwise_distance(o1, o3) ** 2

            tripe_loss = criterion((o1), (o2), (o3))
            # contrastive_loss_pos = criterion_2(
            #     o1, o2, torch.ones(samples.shape[0]).to(device)
            # )
            # loss = 0.6 * tripe_loss + 0.4 * contrastive_loss_pos
            loss = tripe_loss

            running_loss += loss.item()

            valid_count += 1
            if int(valid_count) % 100 == 0:
                log.info(
                    f"valid {valid_count}/{total_len}:{((valid_count ) / (total_len) * 100.0 ):.2f}% in one round...\n"
                )

            del valid_pos_samples
            del valid_neg_samples
            del valid_labels
            del o1, o2, o3
            torch.cuda.empty_cache()

            file_name, file_ext = os.path.splitext(os.path.basename(path[0]))
            all_master_img_names.append(file_name)
            flatten_samples, flatten_labels = flatten_data(samples, labels, "cpu")
            all_labels.append(flatten_labels.to("cpu").tolist())
            all_imgs.append(flatten_samples.to("cpu"))

            all_predicted_dis.append(
                [pos_dis.to("cpu").tolist()[0], neg_dis.to("cpu").tolist()[0]]
            )

    return (
        running_loss / valid_count,
        all_labels,
        all_imgs,
        all_master_img_names,
        all_predicted_dis,
    )
