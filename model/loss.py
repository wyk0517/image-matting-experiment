import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

epsilon = 1e-6
epsilon_sqr = epsilon ** 2


class point_OHEM_loss(nn.Module):
    def __init__(self, stage, radio=0.7):
        super(point_OHEM_loss, self).__init__()
        self.radio = radio
        self.stage = stage

    def forward(self, **input):
        if self.stage == 0:
            loss = point_overall_loss(input['image'], input['alpha'], input['raw_alpha_pred'], input['trimap'],
                                     input['fg'], input['bg'], self.radio)
        elif self.stage == 1:
            loss = point_alpha_prediction_loss(input['alpha'], input['refine_alpha_pred'], input['trimap'], self.radio)
        elif self.stage == 2:
            loss = input['w'] * point_overall_loss(input['image'], input['alpha'], input['raw_alpha_pred'],
                                                  input['trimap'],
                                                  input['fg'], input['bg'], self.radio) + \
                   (1 - input['w']) * point_alpha_prediction_loss(input['alpha'], input['refine_alpha_pred'],
                                                                 input['trimap'], self.radio)
        else:
            assert ()

        return loss


def point_alpha_prediction_loss(alpha, pred, trimap, radio):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size=trimap.size()[0]
    alpha = alpha / 255
    diff = alpha - pred
    diff = diff * unknown_area
    diff = torch.sqrt(diff**2 + epsilon_sqr)

    tot_loss = torch.tensor(0).float().cuda()
    for i in range(batch_size):
        pred_diff = diff[i]
        pred_num = torch.sum(unknown_area[i])
        pred_diff = pred_diff.view(-1)
        pred_num = int(pred_num*radio)
        loss = torch.topk(pred_diff, pred_num)[0]
        loss = torch.sum(loss) / (pred_num + epsilon)
        tot_loss = tot_loss + loss

    return tot_loss / batch_size


def point_compositional_loss(img, pred, fg, bg, trimap, radio):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size = trimap.size()[0]
    unknown_area_trip = torch.cat((unknown_area, unknown_area, unknown_area), 1)
    pred_img = fg * pred + (1 - pred) * bg
    diff = img - pred_img
    diff = diff * unknown_area_trip
    diff = torch.sqrt(diff**2 + epsilon_sqr)
    diff = diff[:, 0, :, :] + diff[:, 1, :, :] + diff[:, 2, :, :]

    tot_loss = torch.tensor(0).float().cuda()

    for i in range(batch_size):
        pred_diff = diff[i]
        pred_num = torch.sum(unknown_area[i])
        # print(pred_diff.size())
        pred_diff = pred_diff.view(-1)
        pred_num = int(pred_num * radio)
        loss = torch.topk(pred_diff, pred_num)[0]
        loss = torch.sum(loss) / (pred_num + epsilon)
        tot_loss = tot_loss + loss

    return tot_loss / batch_size


def point_overall_loss(img, alpha, pred, trimap, fg, bg, radio):
    w = 0.5
    return w * point_alpha_prediction_loss(alpha, pred, trimap, radio) + (1 - w) * point_compositional_loss(img, pred, fg, bg, trimap, radio)


class OHEM_loss(nn.Module):
    def __init__(self, stage, radio=0.7):
        super(OHEM_loss, self).__init__()
        self.radio = radio
        self.stage = stage

    def forward(self, **input):
        if self.stage == 0:
            loss = overall_loss_ohem(input['image'], input['alpha'], input['raw_alpha_pred'], input['trimap'],
                                     input['fg'], input['bg'])
        elif self.stage == 1:
            loss = alpha_prediction_loss_ohem(input['alpha'], input['refine_alpha_pred'], input['trimap'])
        elif self.stage == 2:
            loss = input['w'] * overall_loss_ohem(input['image'], input['alpha'], input['raw_alpha_pred'], input['trimap'],
                                         input['fg'], input['bg']) + \
                   (1 - input['w']) * alpha_prediction_loss_ohem(input['alpha'], input['refine_alpha_pred'], input['trimap'])
        else:
            assert ()

        loss_num = int(loss.size()[0] * self.radio)
        new_loss = torch.topk(loss, loss_num, dim=0)[0]
        # print(new_loss.size(), loss_num, loss.size()[0], self.radio)
        return torch.mean(new_loss)


def nll_loss(output, target):
    return F.nll_loss(output, target)


def alpha_prediction_loss_ohem(alpha, pred, trimap):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size=trimap.size()[0]
    num_pixels = torch.sum(unknown_area.view(batch_size,-1), 1)
    alpha = alpha / 255
    diff = alpha - pred
    diff = diff * unknown_area
    loss = torch.sum(torch.sqrt(diff ** 2 + epsilon_sqr).view(batch_size,-1), 1) / (num_pixels + epsilon)
    return loss


def compositional_loss_ohem(img, pred, fg, bg, trimap):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size=trimap.size()[0]
    unknown_area = torch.cat((unknown_area, unknown_area, unknown_area), 1)
    num_pixels = torch.sum(unknown_area.view(batch_size,-1), 1)
    pred_img = fg * pred + (1 - pred) * bg
    diff = img - pred_img
    diff = diff * unknown_area
    loss = torch.sum(torch.sqrt(diff ** 2 + epsilon_sqr).view(batch_size,-1), 1) / (num_pixels + epsilon) / 255
    return loss


def overall_loss_ohem(img, alpha, pred, trimap, fg, bg):
    w = 0.5
    return w * alpha_prediction_loss_ohem(alpha, pred, trimap) + (1 - w) * compositional_loss_ohem(img, pred, fg, bg,
                                                                                                   trimap)


def alpha_prediction_loss(alpha, pred, trimap):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size=trimap.size()[0]
    #num_pixels = torch.sum(unknown_area.view(batch_size,-1), 1)
    num_pixels=torch.sum(unknown_area)
    alpha = alpha / 255
    diff = alpha - pred
    diff = diff * unknown_area
    #loss = torch.sum(torch.sqrt(diff ** 2 + epsilon_sqr).view(batch_size,-1), 1) / (num_pixels + epsilon)
    #return torch.mean(loss)
    loss=torch.sum(torch.sqrt(diff**2+epsilon_sqr))/(num_pixels + epsilon)
    return loss


def compositional_loss(img, pred, fg, bg, trimap):
    unknown_area = torch.zeros(trimap.shape).cuda()
    unknown_area[trimap == 128] = 1
    batch_size = trimap.size()[0]
    unknown_area = torch.cat((unknown_area, unknown_area, unknown_area), 1)
    #num_pixels = torch.sum(unknown_area.view(batch_size, -1), 1)
    num_pixels=torch.sum(unknown_area)
    pred_img = fg * pred + (1 - pred) * bg 
    diff = img - pred_img
    diff = diff * unknown_area
    #loss = torch.sum(torch.sqrt(diff ** 2 + epsilon_sqr).view(batch_size, -1), 1) / (num_pixels + epsilon) / 255
    #return torch.mean(loss)
    loss=torch.sum(torch.sqrt(diff**2+epsilon_sqr))/(num_pixels+epsilon)/255
    return loss


def overall_loss(img, alpha, pred, trimap, fg, bg):
    w = 0.5
    return w * alpha_prediction_loss(alpha, pred, trimap) + (1 - w) * compositional_loss(img, pred, fg, bg, trimap)
