import torch
import os
import numpy as np
import random
import cv2
import torch.nn as nn
import torchvision
import torchvision.models as models


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_final_output(pred_alpha, trimap):
    trimap = trimap.cpu().numpy()[0, 0, :, :]
    trimap = trimap.reshape((trimap.shape[0], trimap.shape[1], 1))
    mask = 1 - (np.equal(trimap, 0).astype(np.float32) + np.equal(trimap, 255).astype(np.float32))
    return (1 - mask) * trimap + mask * pred_alpha


def gen_trimap(alpha):
    k_size = random.choice(range(3, 7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=np.random.randint(1, 20))
    # eroded = cv2.erode(alpha, kernel)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[alpha >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


# image gradient
def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad=cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad


def architecture_transform(model, backbone, num_layers):
    # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
    # features = vgg.features   # get the features part
    index = -1
    backbone_parameters = []
    for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
        if isinstance(k, nn.Conv2d):
            backbone_parameters.append(k)

    for i, k in enumerate(model):  # transform the para to the model
        if isinstance(k, nn.Conv2d):
            index += 1
            if index > num_layers:
                break
            #if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
            #    break
            para = backbone_parameters[index]
            if index == 0:
                weight = para.weight
                #         if transposed:
                #             self.weight = Parameter(torch.Tensor(
                #                 in_channels, out_channels // groups, *kernel_size))
                #         else:
                #             self.weight = Parameter(torch.Tensor(
                #                 out_channels, in_channels // groups, *kernel_size))
                #         if bias:
                #             self.bias = Parameter(torch.Tensor(out_channels))
                zero_channel = torch.zeros(weight.size(0), 1, weight.size(2), weight.size(3))  # input is trimap=1
                # print(zero_channel.shape)
                weight = torch.cat((weight, zero_channel), dim=1)  # input channel = 4
                k.weight = torch.nn.Parameter(weight)
            else:
                k.weight = torch.nn.Parameter(para.weight)
    return model


def get_layers(model):
    index = -1
    for i, k in enumerate(model):
        if isinstance(k, nn.Conv2d):
            index = index + 1
    return index
