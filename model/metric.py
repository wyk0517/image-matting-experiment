# -*- coding: utf-8 -*-
import torch
import numpy as np


"""
def my_metric(output, target): # 准确率
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
"""
epsilon = 1e-7
epsilon_sqr = epsilon**2


def MSE(pred, alpha, trimap):
    with torch.no_grad():
        batch = alpha.size()[0]
        alpha = alpha / 255
        error_map = pred-alpha
        mask = torch.zeros(trimap.shape).cuda()
        mask[trimap == 128] = 1
        num_pixel = torch.sum(mask.view(batch, -1), 1)
        error_map = error_map**2 * mask
        loss = torch.sum(error_map.view(batch, -1), 1) / num_pixel
        loss = torch.mean(loss)

    return loss


def SAD(pred, alpha, trimap):
    with torch.no_grad():
        batch = alpha.size()[0]
        alpha = alpha / 255
        error_map = torch.abs(pred - alpha)
        mask = torch.zeros(trimap.shape).cuda()
        mask[trimap == 128] = 1
        num_pixel = torch.sum(mask.view(batch, -1), 1)
        error_map = error_map * mask
        loss = torch.sum(error_map.view(batch, -1), 1)
        loss = torch.mean(loss) / 1000

        # the loss is scaled by 1000 due to the large images used in our experiment.
        # loss = loss / 1000
        # print('sad_loss: ' + str(loss))
    return loss
