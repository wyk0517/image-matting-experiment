# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models


def architecture_transform(model, backbone=models.vgg16(pretrained=True)):
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
            if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
                break
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
                zero_channel = torch.zeros(weight.size(0), 1, weight.size(2), weight.size(3)) # input is trimap=1
                # print(zero_channel.shape)
                weight = torch.cat((weight, zero_channel), dim=1)  # input channel = 4
                k.weight = torch.nn.Parameter(weight)
            else:
                k.weight = torch.nn.Parameter(para.weight)
    return model


class Encoder_VGG_transform(nn.Module):
    # input is [batch_size, 4, 320, 320]
    def __init__(self):
        super(Encoder_VGG_transform, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1,padding=1),  # feature map is not vary after conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # after 5 pooling the rest feature map is 7*7(for 224*224 input, and for this task, the input is 320,
        # rest feature is 10*10, to reference the decoder config paper mentioned, the kernel size sets 1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, stride=1),  # the first FC layer transformed conv layer
            nn.ReLU(inplace=True)  # so the feature map is [batch_size, 4096,10, 10]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.init_weights()

    def forward(self, x):
        # get the max_val and max_index for MaxUnpool
        orig_index = []
        x = self.conv1(x)
        x, index = self.maxpool(x)
        orig_index.append(index)
        x = self.conv2(x)
        x, index = self.maxpool(x)
        orig_index.append(index)
        x = self.conv3(x)
        x, index = self.maxpool(x)
        orig_index.append(index)
        x = self.conv4(x)
        x, index = self.maxpool(x)
        orig_index.append(index)
        x = self.conv5(x)
        x, index = self.maxpool(x)
        orig_index.append(index)
        x = self.conv6(x)
        # need to build a "stack"
        orig_index.reverse()
        return x, orig_index

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Decoder_origal(nn.Module):  #  the origal paper's decoder
    def __init__(self):
        super(Decoder_origal, self).__init__()
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, 1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv_predict = nn.Conv2d(64, 1, 5, stride=1, padding=2)
        self.unpooling = nn.MaxUnpool2d(2, 2)
        self.init_weights()

    def forward(self, x, orig_index):
        # x [batch_size, 4096, 10, 10]
        x = self.deconv6(x)  # [batch_size, 512, 10, 10]

        x = self.unpooling(x, orig_index[0])  # [ , 512, 20, 20]
        x = self.deconv5(x)  # [ , 512, 20, 20]

        x = self.unpooling(x, orig_index[1])  # [ , 512, 40, 40]
        x = self.deconv4(x)  # [ , 256, 40, 40]

        x = self.unpooling(x, orig_index[2])  # [, 256, 80, 80]
        x = self.deconv3(x)  # [, 128, 80, 80]

        x = self.unpooling(x, orig_index[3])  # [, 128, 160, 160]
        x = self.deconv2(x)  # [, 64, 160, 160]

        x = self.unpooling(x, orig_index[4])  # [, 64, 320, 320]
        x = self.deconv1(x)  # [, 64, 320, 320]

        x = self.deconv_predict(x)  # [, 1, 320, 320]
        x = F.sigmoid(x)  # scale -> [0, 1] or use x / (max-min) to normalize
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class refinement(nn.Module):
    def __init__(self):
        super(refinement, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_predict = nn.Sequential(
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )

    def forward(self, x, raw_alpha_pred):
        x = torch.cat((x, raw_alpha_pred*255), dim=1)  # scale [0-255]
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_predict(x)
        x = F.sigmoid(x + raw_alpha_pred)
        return x


#################################################
# stage = 0: training the encoder-decoder stage
# stage = 1: training the refinement stage
# stage = 2: fine-tuning the whole model
#################################################
class DIM(BaseModel):
    def __init__(self, backbone, encoder, decoder, refinement, stage):
        super(DIM, self).__init__()
        # self.features = torch.nn.DataParallel(self.features).cuda()
        self.encoder = eval(encoder)  # get from config.jsoon
        self.encoder = architecture_transform(self.encoder, eval(backbone))
        self.decoder = eval(decoder)
        self.stage = stage
        if self.stage == 1:
            # for p in self.parameters():
            #   p.requires_grad=False
            for p in self.encoder.parameters():
                p.requires_grad=False
            for p in self.decoder.parameters():
                p.requires_grad=False

        if self.stage >= 1:
            self.refinement = eval(refinement)

    def forward(self, x):   # x contain image and trimap [batch_size, 4, 320, 320]
        orig_image = x[:, 0:3, :, :]
        x, orig_index = self.encoder(x)
        raw_alpha_pred = self.decoder(x, orig_index)
        # print(raw_alpha_pred)
        # input()
        if self.stage == 0:
            # print("***************************************************")
            return raw_alpha_pred
        elif self.stage >= 1:
            # print("***************************************************")
            refine_alpha_pred = self.refinement(orig_image, raw_alpha_pred)
            return raw_alpha_pred, refine_alpha_pred
        else:
            pass
