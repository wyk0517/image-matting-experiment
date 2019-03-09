import torch.nn as nn
import torch
import numpy as np
import math


class AAF_Loss(nn.Module):
    def __init__(self, step, total_step, kl_lamda1=1.0, kl_lamda2=1.0, kl_margin=3.0, num_class=2):
        super(AAF_Loss, self).__init__()
        # 3 表示3个size
        self.w_edge = torch.zeros((1, 1, 1, num_class, 1, 3)).float().cuda()
        self.w_not_edge = torch.zeros((1, 1, 1, num_class, 1, 3)).float().cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.num_class = num_class
        self.step = step
        self.total_step = total_step

        self.kl_lamda1 = kl_lamda1
        self.kl_lamda2 = kl_lamda2
        self.kl_margin = kl_margin

    def forward(self, pred, gt):
        # 权重，每个类加和等于1
        self.w_edge = self.softmax(self.w_edge)
        self.w_not_edge = self.softmax(self.w_not_edge)

        # one hot 向量
        one_hot = self.one_hot(gt)
        # 3*3
        eloss_1, neloss_1 = self.compute_aaf_loss(gt, one_hot,
                                                  pred, 1,
                                                  self.kl_margin,
                                                  self.w_edge[..., 0], self.w_not_edge[..., 0])
        # 5*5
        eloss_2, neloss_2 = self.compute_aaf_loss(gt, one_hot,
                                                  pred, 2,
                                                  self.kl_margin,
                                                  self.w_edge[..., 1], self.w_not_edge[..., 1])
        # 7*7
        eloss_3, neloss_3 = self.compute_aaf_loss(gt, one_hot,
                                                  pred, 3,
                                                  self.kl_margin,
                                                  self.w_edge[..., 2], self.w_not_edge[..., 2])

        # decay
        dec = math.pow(10.0, -self.step / self.total_step)
        # 三个尺寸相加
        aaf = torch.mean(eloss_1).item() * self.kl_lamda1 * dec
        aaf += torch.mean(eloss_2).item() * self.kl_lamda1 * dec
        aaf += torch.mean(eloss_3).item() * self.kl_lamda1 * dec
        aaf += torch.mean(neloss_1).item() * self.kl_lamda2 * dec
        aaf += torch.mean(neloss_2).item() * self.kl_lamda2 * dec
        aaf += torch.mean(neloss_3).item() * self.kl_lamda2 * dec
        return aaf

    # 生成one hot 向量
    def one_hot(self, gt):
        label = gt.clone().cpu()
        if len(label.size()) == 3:
            label = torch.unsqueeze(label, dim=-1)
        n, h, w, _ = label.size()
        c = self.num_class

        # 全1矩阵
        ones = torch.zeros((n, h, w, c))
        ones += 1

        label = label.long()
        one_hot = torch.zeros((n, h, w, c)).scatter_(dim=3, index=label, src=ones).float().cuda()
        return one_hot

    # 得到八个方向的边，label不同连边
    def get_edge(self, label, size):
        # N*H*W*1 to N*H*W
        label = torch.unsqueeze(label, dim=-1)
        n, h, w, c = label.size()
        # padding
        label_pad = torch.zeros((n, h + 2 * size, w + 2 * size, c)).float().cuda()
        label_pad[:, size:size + h, size:size + w, :] = label
        # 计算八个方向的边
        edge_groups = []
        for i in range(0, 2 * size + 1, size):
            for j in range(0, 2 * size + 1, size):
                if i == size and j == size:
                    continue
                label_neighbor = label_pad[:, i:i + h, j:j + h, :]
                edge = torch.zeros((n, h, w, c)).float().cuda()
                edge[label_neighbor != label] = 1
                edge_groups.append(edge)
        # 扩展一维到n*h*w*1*1
        edge_groups = [torch.unsqueeze(i, dim=-1) for i in edge_groups]
        # N*H*W*1*8
        edge = torch.cat(edge_groups, dim=-1)
        return edge

    # 把预测值扩展到八个方向
    def move_pred(self, pred, size):
        # pred N*H*W*C
        n, h, w, c = pred.size()
        # padding
        pred_pad = torch.zeros((n, h + 2 * size, w + 2 * size, c)).float().cuda()
        pred_pad[:, size:size + h, size:size + w, :] = pred
        # 叠加八个方向
        pred_groups = []
        for i in range(0, 2 * size + 1, size):
            for j in range(0, 2 * size + 1, size):
                if i == size and j == size:
                    continue
                pred_neighbor = pred_pad[:, i:i + h, j:j + h, :]
                pred_groups.append(pred_neighbor)
        # 扩展一维到n*h*w*c*1
        pred_groups = [torch.unsqueeze(i, dim=-1) for i in pred_groups]
        # N*H*W*C*8
        output = torch.cat(pred_groups, dim=-1)
        return output

    # 得到对应下标
    def get_index(self, x):
        n = x.size(0)
        tmp = torch.arange(0, n).float().cuda()
        tmp = tmp * x
        out = tmp[tmp > 0]
        return out

    # 计算loss的过程
    def compute_aaf_loss(self, label, one_hot, pred, size, KL_margin, w_edge, w_not_edge):
        """
        :param label: N*H*W*1  GTlabel
        :param one_hot:   gt的one hot向量
        :param pred:  N*H*W*C  model output after a softmax
        :param size:  affinity field is 2*size+1
        :param KL_margin:  阈值
        :param w_edge:  权重
        :param w_not_edge: 权重
        :return:
        """

        # N*H*W*1 to N*H*W
        label = torch.squeeze(label, dim=-1)

        # 连边 N*H*W*1*8
        edge = self.get_edge(label, size)
        edge_index = self.get_index(edge.view(-1)).long()  # 一维

        not_edge = 1 - edge
        not_edge_index = self.get_index(not_edge.view(-1)).long()

        # 把pred移动八个方向  N*H*W*C*8
        pred_paired = self.move_pred(pred, size)
        # pred扩展一维N*H*W*C*1，和pred_paired对齐
        pred = torch.unsqueeze(pred, dim=-1)

        # 指定概率区间的最大值和最小值
        min_prob = 1e-4
        max_prob = 1
        neg_pred = torch.clamp(pred, min_prob, max_prob)
        neg_pred_paired = torch.clamp(pred_paired, min_prob, max_prob)
        pred = torch.clamp(pred, min_prob, max_prob)
        pred_paired = torch.clamp(pred_paired, min_prob, max_prob)

        # 计算KL散度
        kl = pred_paired * torch.log(pred_paired / pred)
        kl += neg_pred_paired * torch.log(neg_pred_paired / neg_pred)
        # edge_loss = torch.max(0, KL_margin - kl)  不知道有没有简单的实现这个max的方法，下面用了一个mask实现这个
        mask = torch.zeros(kl.size()).float().cuda()
        mask[KL_margin - kl > 0] = 1
        edge_loss = (KL_margin - kl) * mask
        not_edge_loss = kl

        one_hot = torch.unsqueeze(one_hot, dim=-1)  # N*H*W*C*1
        w_edge = torch.sum(w_edge * one_hot, dim=3, keepdim=True)  # N*H*W*1*1 从类别那一维相加
        w_not_edge = torch.sum(w_not_edge * one_hot, dim=3, keepdim=True)

        # 乘以权重
        edge_loss *= w_edge
        not_edge_loss *= w_not_edge

        # 取对应有边和无边的下标
        not_edge_loss = not_edge_loss.view(-1)
        not_edge_loss = torch.gather(not_edge_loss, dim=0, index=not_edge_index)

        edge_loss = edge_loss.view(-1)
        edge_loss = torch.gather(edge_loss, dim=0, index=edge_index)
        return edge_loss, not_edge_loss


n, h, w = 10, 320, 320
pred = torch.rand((n, h, w, 1)).cuda()
gt = torch.rand((n, h, w, 1)).cuda()
gt[gt > 0.5] = 1
gt[gt != 1] = 0
print(torch.sum(gt))
aaf = AAF_Loss(12304, 20000)
out = aaf(pred, gt)
print(out)

out = aaf(pred, gt)
print(out)
