import torch.nn.functional as nnF
import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, labels):
        # 计算欧式距离
        euclidean_distance = nnF.pairwise_distance(output1, output2, keepdim=True)

        # 计算损失
        loss = torch.mean(
            labels * torch.pow(euclidean_distance, 2)
            + (1 - labels)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss


class CustomTripletMarginLossWithSwap(nn.Module):
    def __init__(self, margin=1.0, p=2, reduction="mean", swap=False):
        """
        :param margin: 正负样本之间的最小距离差距
        :param p: 计算距离的范数，默认使用欧式距离 (p=2)
        :param reduction: 'mean' 或 'sum'，决定损失如何在批次上汇总
        :param swap: 是否使用负样本与正样本距离交换
        """
        super(CustomTripletMarginLossWithSwap, self).__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
        self.swap = swap

    def forward(self, anchor, positive, negative):
        # 计算 anchor-positive 和 anchor-negative 的欧式距离
        positive_distance = nnF.pairwise_distance(anchor, positive, p=self.p) ** 2
        negative_distance = nnF.pairwise_distance(anchor, negative, p=self.p) ** 2

        # 交换机制：计算 positive 与 negative 之间的距离
        if self.swap:
            # 计算 positive-negative 之间的距离 (p和n作为anchor进行的计算)
            pn_distance = nnF.pairwise_distance(positive, negative, p=self.p) ** 2
            # 取 negative_distance 和 pn_distance 的最小值
            negative_distance = torch.min(negative_distance, pn_distance)

        # 计算 Triplet 损失
        losses = nnF.relu(positive_distance - negative_distance + self.margin)

        # 按照 reduction 规则汇总损失
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses
