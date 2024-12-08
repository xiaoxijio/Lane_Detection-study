import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)  # 权重 样本预测误差越小权重越小 误差越大权重越大
        log_score = F.log_softmax(logits, dim=1)  # 交叉熵
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape  # (B, 201, 18, 4)
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])  # 两个相邻的车道点概率接近
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()

    def forward(self, x):
        n, dim, num_rows, num_cols = x.shape  # (B, 201, 18, 4) 200个位置的概率+最后一个判断有没有车道线
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)  # (B, 200, 18, 4)  最后一个去掉了
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)  # 求期望(直接取最大值没办法反向传播)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])  # 两个相邻的车道点位置接近

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss
