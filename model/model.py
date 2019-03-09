# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
import torch.utils.model_zoo as model_zoo
# from utils import architecture_transform as architecture_transform
from utils import get_layers as get_layers
import torchvision.models as models


#################################################
# stage = 0: training the encoder-decoder stage
# stage = 1: training the refinement stage
# stage = 2: fine-tuning the whole model
#################################################

#############################################
#
# Learning Affinity via Spatial Propagation Networks
#
##############################################
def norm_D(x, y, z):
    sum = x.abs() + y.abs() + z.abs()
    mask = sum.ge(1).float  # sum>1 return true
    tmp = torch.add(-mask, 1)
    x = tmp * x + mask * torch.div(x, sum)
    y = tmp * y + mask * torch.div(y, sum)
    z = tmp * z + mask * torch.div(z, sum)
    return x, y, z

class affinity_module(BaseModel):
    def __init__(self, backbone, stage):
        super(affinity_module, self).__init__()
        self.vgg = VGG_transform_net(backbone, 0)
        self.stage = stage
        if self.stage == 1:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.spn = SPN_net(backbone)

    def forward(self, x):
        orig_img = x[:, 0:3, :, :]
        coarse = self.vgg(x)
        #for param in self.vgg.parameters():
        #    print(param.requires_grad)
        if self.stage == 0:
            return coarse
        output = self.spn(orig_img, coarse)
        return output

class SPN_net(BaseModel):
    # input is [batch_size, 4, 320, 320]
    def __init__(self, backbone):
        super(SPN_net, self).__init__()
        ##################
        # encoder
        ##################
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # feature map is not vary after conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
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
        encoder_model = self.modules()
        num_layers = get_layers(encoder_model)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, stride=1),  # the first FC layer transformed conv layer
            nn.ReLU(inplace=True)  # so the feature map is [batch_size, 4096,10, 10]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        ##################
        # decoder
        ##################
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, 1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16 * 3 * 4, 3, stride=1, padding=1),  # C*3*4 three way four direction
            nn.ReLU()  # to (0,1)
        )
        self.unpooling = nn.MaxUnpool2d(2, 2)
        ##############
        # affinity
        ##############

        self.refine_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.refine_conv2 = nn.Sequential(
            nn.Conv2d(16 * 3 * 4, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        ###################
        # weight init
        ###################
        self.init_weights()
        #architecture_transform(self.modules(), eval(backbone), num_layers)

    def forward(self, x, coarse):
        # get the max_val and max_index for MaxUnpool
        ##################
        # encoder
        ##################
        # x [batch_size, 4, 320, 320]
        orig_img = x
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
        ##################
        # decoder
        ##################
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
        x = self.deconv1(x)  # [, 16*3*4, 320, 320]
        # finish generate weight

        # process coarse output
        coarse = coarse * 255  # turn [0,1] to [0,255]
        coarse = self.refine_conv1(coarse)

        # four direction
        channel_each_direct = x.size()[1] // 4  # x.size()  n,c,h,w
        D1 = x[:, 0:channel_each_direct, :, :]
        D2 = x[:, channel_each_direct:2 * channel_each_direct, :, :]
        D3 = x[:, 2 * channel_each_direct:3 * channel_each_direct, :, :]
        D4 = x[:, 3 * channel_each_direct:4 * channel_each_direct, :, :]

        # three way
        channel_each_way = channel_each_direct // 3
        D1_w1 = D1[:, 0:channel_each_way, :, :]
        D1_w2 = D1[:, channel_each_way:2 * channel_each_way, :, :]
        D1_w3 = D1[:, 2 * channel_each_way:3 * channel_each_way, :, :]
        # |w1| + |w2| + |w3| should <=1
        D1_w1, D1_w2, D1_w3 = norm_D(D1_w1, D1_w2, D1_w3)

        # left to right
        Propagator = GateRecurrent2dnoind(True, False)
        output = Propagator.forward(coarse, D1_w1, D1_w2, D1_w3)

        D2_w1 = D2[:, 0:channel_each_way, :, :]
        D2_w2 = D2[:, channel_each_way:2 * channel_each_way, :, :]
        D2_w3 = D2[:, 2 * channel_each_way:3 * channel_each_way, :, :]
        # |w1| + |w2| + |w3| should <=1
        D2_w1, D2_w2, D2_w3 = norm_D(D2_w1, D2_w2, D2_w3)

        # right to left
        Propagator = GateRecurrent2dnoind(True, True)
        output = Propagator.forward(output, D1_w1, D1_w2, D1_w3)

        D3_w1 = D3[:, 0:channel_each_way, :, :]
        D3_w2 = D3[:, channel_each_way:2 * channel_each_way, :, :]
        D3_w3 = D3[:, 2 * channel_each_way:3 * channel_each_way, :, :]
        # |w1| + |w2| + |w3| should <=1
        D3_w1, D3_w2, D3_w3 = norm_D(D3_w1, D3_w2, D3_w3)

        # top to bottom
        Propagator = GateRecurrent2dnoind(False, False)
        output = Propagator.forward(output, D1_w1, D1_w2, D1_w3)

        D4_w1 = D4[:, 0:channel_each_way, :, :]
        D4_w2 = D4[:, channel_each_way:2 * channel_each_way, :, :]
        D4_w3 = D4[:, 2 * channel_each_way:3 * channel_each_way, :, :]
        # |w1| + |w2| + |w3| should <=1
        D4_w1, D4_w2, D4_w3 = norm_D(D4_w1, D4_w2, D4_w3)

        # bottom to to
        Propagator = GateRecurrent2dnoind(False, True)
        output = Propagator.forward(output, D1_w1, D1_w2, D1_w3)

        output = self.refine_conv2(output)

        return output

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

    def architecture_transform(self, backbone, num_layers):
        # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
        # features = vgg.features   # get the features part
        index = -1
        backbone_parameters = []
        for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
            if isinstance(k, nn.Conv2d):
                backbone_parameters.append(k)

        for i, k in enumerate(self.modules()):  # transform the para to the model
            if isinstance(k, nn.Conv2d):
                index += 1
                if index > num_layers:
                    break
                # if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
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

#####################################################
#
# rcf-net based on vgg
#
###################################################
def rcf_conv3x3(in_ch, out_ch, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, dilation=dilation)


def rcf_conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class rcf_vgg(BaseModel):
    def __init__(self, stage, backbone):
        super(rcf_vgg, self).__init__()
        self.stage = stage
        self.conv1_1 = rcf_conv3x3(4, 64)
        self.conv1_2 = rcf_conv3x3(64, 64)

        self.conv2_1 = rcf_conv3x3(64, 128)
        self.conv2_2 = rcf_conv3x3(128, 128)

        self.conv3_1 = rcf_conv3x3(128, 256)
        self.conv3_2 = rcf_conv3x3(256, 256)
        self.conv3_3 = rcf_conv3x3(256, 256)

        self.conv4_1 = rcf_conv3x3(256, 512)
        self.conv4_2 = rcf_conv3x3(512, 512)
        self.conv4_3 = rcf_conv3x3(512, 512)

        self.conv5_1 = rcf_conv3x3(512, 512, stride=1, padding=2, dilation=2)
        self.conv5_2 = rcf_conv3x3(512, 512, stride=1, padding=2, dilation=2)
        self.conv5_3 = rcf_conv3x3(512, 512, stride=1, padding=2, dilation=2)
        encoder_model = self.modules()
        num_layers = get_layers(encoder_model)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, 1, ceil_mode=True)

        self.conv1_down = rcf_conv1x1(64, 21)
        self.conv2_down = rcf_conv1x1(128, 21)
        self.conv3_down = rcf_conv1x1(256, 21)
        self.conv4_down = rcf_conv1x1(512, 21)
        self.conv5_down = rcf_conv1x1(512, 21)

        self.score1 = rcf_conv1x1(21, 1)
        self.score2 = rcf_conv1x1(21, 1)
        self.score3 = rcf_conv1x1(21, 1)
        self.score4 = rcf_conv1x1(21, 1)
        self.score5 = rcf_conv1x1(21, 1)
        self.score_final = rcf_conv1x1(5, 1)

        if self.stage == 1:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True
        ##############
        # refinement
        ##############
        if self.stage == 1 or self.stage == 2:
            self.refine_conv1 = nn.Sequential(
                nn.Conv2d(4, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv_predict = nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=1, padding=1)
            )
        self.init_weight()
        self.architecture_transform(eval(backbone), num_layers)

    def forward(self, x):
        orig_img = x[:,0:3,:,:]
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_down(conv1_1)
        conv1_2_down = self.conv1_down(conv1_2)
        conv2_1_down = self.conv2_down(conv2_1)
        conv2_2_down = self.conv2_down(conv2_2)
        conv3_1_down = self.conv3_down(conv3_1)
        conv3_2_down = self.conv3_down(conv3_2)
        conv3_3_down = self.conv3_down(conv3_3)
        conv4_1_down = self.conv4_down(conv4_1)
        conv4_2_down = self.conv4_down(conv4_2)
        conv4_3_down = self.conv4_down(conv4_3)
        conv5_1_down = self.conv5_down(conv5_1)
        conv5_2_down = self.conv5_down(conv5_2)
        conv5_3_down = self.conv5_down(conv5_3)

        so1_out = self.score1(conv1_1_down + conv1_2_down)
        so2_out = self.score2(conv2_1_down + conv2_2_down)
        so3_out = self.score3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score5(conv5_1_down + conv5_2_down + conv5_3_down)

        upsample2 = F.upsample_bilinear(so2_out, (img_H, img_W))
        upsample3 = F.upsample_bilinear(so3_out, (img_H, img_W))
        upsample4 = F.upsample_bilinear(so4_out, (img_H, img_W))
        upsample5 = F.upsample_bilinear(so5_out, (img_H, img_W))

        f_cat = torch.cat((so1_out, upsample2, upsample3, upsample4, upsample5), dim=1)
        f_use = self.score_final(f_cat)
        results = [so1_out, upsample2, upsample3, upsample4, upsample5, f_use]
        results = [F.sigmoid(r) for r in results]
        raw_alpha_pred = results[-1]

        if self.stage == 0:
            # print(raw_alpha_pred.shape)
            return results

        x = torch.cat((orig_img, raw_alpha_pred * 255), dim=1)  # scale [0-255]
        x = self.refine_conv1(x)
        x = self.refine_conv2(x)
        x = self.refine_conv3(x)
        refine_alpha_pred = self.refine_conv_predict(x)
        refine_alpha_pred = F.sigmoid(refine_alpha_pred)
        refine_alpha_pred = (refine_alpha_pred + raw_alpha_pred) / 2
        results.append(refine_alpha_pred)
        return results

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def architecture_transform(self, backbone, num_layers):
        # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
        # features = vgg.features   # get the features part
        index = -1
        backbone_parameters = []
        for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
            if isinstance(k, nn.Conv2d):
                backbone_parameters.append(k)

        for i, k in enumerate(self.modules()):  # transform the para to the model
            if isinstance(k, nn.Conv2d):
                index += 1
                if index > num_layers:
                    break
                # if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
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

models.resnet50()
##################################################
#
# PSP-net based on resnet-50
#
##################################################
'''
Note:
    PSPNet: first conv7k_2s modify conv3k_2s/conv3k_1s/conv3k_1s(3 layers)
    each downsample block: first conv1k_1s modify conv1k_2s; second conv3k_2s modify conv3k_1s(deplab resnet)
    layer1: no downsample
    layer2: downsample
    layer3: no downsample; each block the second conv3x3 modify atros_conv3k_2r
    layer4: no downsample; each block the second conv3x3 modify atros_conv3k_4r
    Note: Resnet no bias,so bias = False

'''


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_atrous(in_planes, out_planes, rate=1, padding=1, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=rate, padding=padding, bias=False)


class PSP_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None):
        '''
        pspnet conv1_3's num_output=128 not 64 so we modify some code
        first_inplanes: only layer1 not same (conv1_3)128 != (layer1-block1-conv1k_1s)64
        '''
        super(PSP_Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_atrous(planes, planes, rate, padding)  # change
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # only first layer1 block in_channel different
        if (first_inplanes != inplanes) and (downsample is not None):
            self.conv1 = conv1x1(first_inplanes, planes, stride)
            self.downsample = nn.Sequential(conv1x1(first_inplanes, planes * self.expansion, stride),
                                            nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class SPP(nn.Module):
    # no bias
    def __init__(self, in_plances, plances, level):
        super(SPP, self).__init__()
        self.in_plances = in_plances
        self.plances = plances
        self.level = level
        self.conv = nn.Sequential(
            nn.Conv2d(in_plances, plances, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(plances),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        feature = []
        orig_feature = x
        for i in range(len(self.level)):
            x = F.adaptive_avg_pool2d(orig_feature, self.level[i])
            # print(x.shape)
            x = self.conv(x)
            x = F.upsample(x, size=(H, W), mode='bilinear', align_corners=True)
            feature.append(x)

        for i in range(len(feature)):
            orig_feature = torch.cat((orig_feature, feature[i]), dim=1)
        return orig_feature


class PSP_Resnet50(BaseModel):
    def __init__(self, stage, block, layers, class_number=1, dropout_rate=0.5):
        super(PSP_Resnet50, self).__init__()
        block = eval(block)
        layers = eval(layers)
        self.inplanes = 64
        self.stage = stage
        self.conv1_1 = conv3x3_bn_relu(4, 64, 2)
        self.conv1_2 = conv3x3_bn_relu(64, 64, 1)
        self.conv1_3 = conv3x3_bn_relu(64, 128, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, 64, layers[0])  # 64 / 256
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)  # 128 / 512
        self.layer3 = self._make_layer(block, 512, 256, layers[2], rate=2, padding=2)  # 256 / 1024
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], rate=4, padding=4)  # 512 / 2048

        self.spp = SPP(2048, 512, [1, 2, 3, 6])
        self.conv5_4 = conv3x3_bn_relu(2048 + 512 * 4, 512)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv6 = nn.Conv2d(512, class_number, 1, 1)

        if self.stage == 1:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True
        ##############
        # refinement
        ##############
        if self.stage == 1 or self.stage == 2:
            self.refine_conv1 = nn.Sequential(
                nn.Conv2d(4, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv_predict = nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=1, padding=1)
            )
        self.init_weight()

    def forward(self, x):

        orig_img = x[:, 0:3, :, :]

        size = x.shape[2:]
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.conv5_4(x)
        x = self.dropout(x)
        x = self.conv6(x)
        raw_alpha_pred = F.upsample(x, size, mode='bilinear', align_corners=True)
        raw_alpha_pred = F.sigmoid(raw_alpha_pred)
        if self.stage == 0:
            # print(raw_alpha_pred.shape)
            return raw_alpha_pred

        x = torch.cat((orig_img, raw_alpha_pred * 255), dim=1)  # scale [0-255]
        x = self.refine_conv1(x)
        x = self.refine_conv2(x)
        x = self.refine_conv3(x)
        refine_alpha_pred = self.refine_conv_predict(x)
        refine_alpha_pred = F.sigmoid(refine_alpha_pred)
        refine_alpha_pred = (refine_alpha_pred + raw_alpha_pred) / 2
        return raw_alpha_pred, refine_alpha_pred

    def _make_layer(self, block, first_inplanes, planes, blocks, rate=1, padding=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # with down stride same
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(first_inplanes, self.inplanes, planes, rate, padding, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, planes, rate, padding))

        return nn.Sequential(*layers)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


#############################################
#
# U-net Based on Deep image matting model
#
##############################################
class VGG_transform_u_net(BaseModel):
    # input is [batch_size, 4, 320, 320]
    def __init__(self, backbone, stage):
        super(VGG_transform_u_net, self).__init__()
        ##################
        # encoder
        ##################
        self.stage = stage
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1,padding=1),  # feature map is not vary after conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
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
        encoder_model = self.modules()
        num_layers=get_layers(encoder_model)
        # after 5 pooling the rest feature map is 7*7(for 224*224 input, and for this task, the input is 320,
        # rest feature is 10*10, to reference the decoder config paper mentioned, the kernel size sets 1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, stride=1),  # the first FC layer transformed conv layer
            nn.ReLU(inplace=True)  # so the feature map is [batch_size, 4096,10, 10]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        ##################
        # decoder
        ##################
        self.deconv6 = nn.Sequential(
            nn.Conv2d(4096, 512, 1, stride=1),
            # nn.Conv2d(4096,512,1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(1024, 512, 5, stride=1, padding=2),
            # nn.Conv2d(512, 512, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(1024, 256, 5, stride=1, padding=2),
            # nn.Conv2d(512, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(512, 128, 5, stride=1, padding=2),
            # nn.Conv2d(256, 128, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(256, 64, 5, stride=1, padding=2),
            # nn.Conv2d(128, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 5, stride=1, padding=2),
            # nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv_predict = nn.Conv2d(64, 1, 5, stride=1, padding=2)
        self.unpooling = nn.MaxUnpool2d(2, 2)
        if self.stage == 1:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True
        ##############
        # refinement
        ##############

        if self.stage == 1 or self.stage == 2:
            self.refine_conv1 = nn.Sequential(
                nn.Conv2d(4, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv_predict = nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=1, padding=1)
            )
        ###################
        # weight init
        ###################
        self.init_weights()
        self.architecture_transform(eval(backbone), num_layers)

    def forward(self, x):
        # get the max_val and max_index for MaxUnpool
        ##################
        # encoder
        ##################
        # x [batch_size, 4, 320, 320]
        orig_img = x[:, 0:3, :, :]
        orig_index = []
        x1 = self.conv1(x)

        x2, index = self.maxpool(x1)
        orig_index.append(index)
        x2 = self.conv2(x2)

        x3, index = self.maxpool(x2)
        orig_index.append(index)
        x3 = self.conv3(x3)

        x4, index = self.maxpool(x3)
        orig_index.append(index)
        x4 = self.conv4(x4)

        x5, index = self.maxpool(x4)
        orig_index.append(index)
        x5 = self.conv5(x5)

        x6, index = self.maxpool(x5)
        orig_index.append(index)
        x6 = self.conv6(x6)

        # need to build a "stack"
        orig_index.reverse()
        ##################
        # decoder
        ##################
        # x [batch_size, 4096, 10, 10]

        x = self.deconv6(x6)  # [batch_size, 512, 10, 10]

        x = self.unpooling(x, orig_index[0],)  # [ , 512, 20, 20]
        x=torch.cat((x,x5),dim=1)
        x = self.deconv5(x)  # [ , 512, 20, 20]

        x = self.unpooling(x, orig_index[1])  # [ , 512, 40, 40]
        x=torch.cat((x,x4),dim=1)
        x = self.deconv4(x)  # [ , 256, 40, 40]

        x = self.unpooling(x, orig_index[2])  # [, 256, 80, 80]
        x=torch.cat((x,x3),dim=1)
        x = self.deconv3(x)  # [, 128, 80, 80]

        x = self.unpooling(x, orig_index[3])  # [, 128, 160, 160]
        x=torch.cat((x,x2),dim=1)
        x = self.deconv2(x)  # [, 64, 160, 160]

        x = self.unpooling(x, orig_index[4])  # [, 64, 320, 320]
        x=torch.cat((x,x1),dim=1)
        x = self.deconv1(x)  # [, 64, 320, 320]

        raw_alpha_pred = self.deconv_predict(x)  # [, 1, 320, 320]
        raw_alpha_pred = F.sigmoid(raw_alpha_pred)  # scale -> [0, 1] or use x / (max-min) to normalize
        if self.stage == 0:
            # print(raw_alpha_pred.shape)
            return raw_alpha_pred

        x = torch.cat((orig_img, raw_alpha_pred * 255), dim=1)  # scale [0-255]
        x = self.refine_conv1(x)
        x = self.refine_conv2(x)
        x = self.refine_conv3(x)
        refine_alpha_pred = self.refine_conv_predict(x)
        refine_alpha_pred = F.sigmoid(refine_alpha_pred)
        refine_alpha_pred = (refine_alpha_pred + raw_alpha_pred) / 2
        return raw_alpha_pred, refine_alpha_pred

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

    def architecture_transform(self, backbone, num_layers):
        # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
        # features = vgg.features   # get the features part
        index = -1
        backbone_parameters = []
        for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
            if isinstance(k, nn.Conv2d):
                backbone_parameters.append(k)

        for i, k in enumerate(self.modules()):  # transform the para to the model
            if isinstance(k, nn.Conv2d):
                index += 1
                if index > num_layers:
                    break
                # if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
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

##########################################
#
# deep image matting paper's model
#
########################################
class VGG_transform_net(BaseModel):
    # input is [batch_size, 4, 320, 320]
    def __init__(self, backbone, stage):
        super(VGG_transform_net, self).__init__()
        ##################
        # encoder
        ##################
        self.stage = stage
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1,padding=1),  # feature map is not vary after conv
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
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
        encoder_model = self.modules()
        num_layers = get_layers(encoder_model)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, stride=1),  # the first FC layer transformed conv layer
            nn.ReLU(inplace=True)  # so the feature map is [batch_size, 4096,10, 10]
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        ##################
        # decoder
        ##################
        self.deconv6 = nn.Sequential(
            nn.Conv2d(4096, 512, 1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(512, 512, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(512, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(256, 128, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.deconv_predict = nn.Conv2d(64, 1, 5, stride=1, padding=2)
        self.unpooling = nn.MaxUnpool2d(2, 2)
        if self.stage == 1:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True
        ##############
        # refinement
        ##############

        if self.stage == 1 or self.stage == 2:
            self.refine_conv1 = nn.Sequential(
                nn.Conv2d(4, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv_predict = nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=1, padding=1)
            )
        ###################
        # weight init
        ###################
        self.init_weights()
        self.architecture_transform(eval(backbone), num_layers)

    def forward(self, x):
        # get the max_val and max_index for MaxUnpool
        ##################
        # encoder
        ##################
        # x [batch_size, 4, 320, 320]
        orig_img = x[:, 0:3, :, :]
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
        ##################
        # decoder
        ##################
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

        raw_alpha_pred = self.deconv_predict(x)  # [, 1, 320, 320]
        raw_alpha_pred = F.sigmoid(raw_alpha_pred)  # scale -> [0, 1] or use x / (max-min) to normalize
        if self.stage == 0:
            # print(raw_alpha_pred.shape)
            return raw_alpha_pred

        x = torch.cat((orig_img, raw_alpha_pred * 255), dim=1)  # scale [0-255]
        x = self.refine_conv1(x)
        x = self.refine_conv2(x)
        x = self.refine_conv3(x)
        refine_alpha_pred = self.refine_conv_predict(x)
        # refine_alpha_pred = F.sigmoid(refine_alpha_pred)
        refine_alpha_pred = F.sigmoid(refine_alpha_pred)
        refine_alpha_pred = (refine_alpha_pred + raw_alpha_pred) / 2
        #refine_alpha_pred = refine_alpha_pred + raw_alpha_pred
        #refine_alpha_pred[ refine_alpha_pred > 1 ] = 1
        return raw_alpha_pred, refine_alpha_pred

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

    def architecture_transform(self, backbone, num_layers):
        # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
        # features = vgg.features   # get the features part
        index = -1
        backbone_parameters = []
        for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
            if isinstance(k, nn.Conv2d):
                backbone_parameters.append(k)

        for i, k in enumerate(self.modules()):  # transform the para to the model
            if isinstance(k, nn.Conv2d):
                index += 1
                if index > num_layers:
                    break
                # if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
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

###################################################################
#
# This model is a depplabv3_plus based on resnet50, but don't use depthwith
# separable convolution because the implement in pytorch is so slow
#
########################################################################
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# upsample x like size of y
def upsample(x, y):
    n, c, h, w = y.shape
    return F.upsample(x, size=(h, w), mode='bilinear', align_corners=True)


class deeplabv3_plus(BaseModel):
    def __init__(self, block, layers, backbone, stage):
        self.inplanes = 64
        super(deeplabv3_plus, self).__init__()
        layers = eval(layers)
        self.backbone = backbone
        self.stage = stage
        self.new_conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # /2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # /2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # /2
        resnet_model = self.modules()
        num_layers = get_layers(resnet_model)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # deeplabv3+ atrous spatial pyramid pooling
        in_chan = 1024
        depth = 256
        atrous_rates = [2, 4, 6]  # case the last feature is 20*20, I choose a small rate
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv1x1 = nn.Sequential(
            nn.Conv2d(in_chan, depth, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
         )
        self.aspp_conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0]),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1]),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_chan, depth, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2]),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.aspp_avg = nn.AdaptiveAvgPool2d(1)
        self.aspp_pooling_conv = nn.Sequential(
            nn.Conv2d(in_chan, depth, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        self.aspp_cat_conv = nn.Sequential(
            nn.Conv2d(depth*5, depth, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True)
        )
        num_classes = 1
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(depth + 48, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv3 = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1)
        if self.stage == 1:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad=False
        ##############################
        #  refinement
        #############################
        if self.stage == 1 or self.stage == 2:
            self.refine_conv1 = nn.Sequential(
                nn.Conv2d(4, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            self.refine_conv_predict = nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=1, padding=1)
            )
        self.init_weights()
        self.architecture_transform(eval(backbone), num_layers)

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        block = eval(block)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def architecture_transform(self, backbone, num_layers):
        # vgg = models.vgg16(pretrained=True)  # contain feature part and classifier part
        # features = vgg.features   # get the features part
        index = -1
        backbone_parameters = []
        for i, k in enumerate(backbone.modules()):  # get the part wanted from the pre-trained model
            if isinstance(k, nn.Conv2d):
                backbone_parameters.append(k)

        for i, k in enumerate(self.modules()):  # transform the para to the model
            if isinstance(k, nn.Conv2d):
                index += 1
                if index > num_layers:
                    break
                # if index == 13:  # because pre-trained model has 13 layer conv_layer, and the encoder has 14 conv_layer
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

    def forward(self, x):
        feature_0 = x
        feature_1 = self.new_conv1(feature_0)  # output_stride = 2
        feature_2 = self.maxpool(feature_1)
        x = self.layer1(feature_2)  # output stride = 4
        feature_3 = self.layer2(x)  # output stride = 8
        feature_4 = self.layer3(feature_3)  # output stride = 16

        # aspp feature 1/16
        aspp_conv1 = self.aspp_conv1x1(feature_4)
        aspp_conv3_1 = self.aspp_conv3x3_1(feature_4)
        aspp_conv3_2 = self.aspp_conv3x3_2(feature_4)
        aspp_conv3_3 = self.aspp_conv3x3_3(feature_4)
        # global avg pooling
        # aspp_pooling = F.avg_pool2d(feature_4, kernel_size=feature_4.size()[2:])
        aspp_pooling = self.aspp_avg(feature_4)
        aspp_pooling = self.aspp_pooling_conv(aspp_pooling)
        aspp_pooling = upsample(aspp_pooling, feature_4)

        # pyramid fusion
        aspp = torch.cat((aspp_conv1, aspp_conv3_1, aspp_conv3_2, aspp_conv3_3, aspp_pooling), dim=1)
        aspp = self.aspp_cat_conv(aspp)

        # cat with low feature 1/4
        low_feat = self.low_level_conv(feature_2)
        aspp_up = upsample(aspp, low_feat)
        cat_feat = torch.cat((aspp_up, low_feat), 1)  # [batch, 48+ depth]

        # pred raw alpha 1/1
        decoder_conv1 = self.decoder_conv1(cat_feat)
        decoder_conv2 = self.decoder_conv2(decoder_conv1)
        decoder_conv3 = self.decoder_conv3(decoder_conv2)
        raw_alpha = upsample(decoder_conv3, feature_0)

        raw_alpha = F.sigmoid(raw_alpha)
        if self.stage == 0:
            return raw_alpha
        # stage2 refine conv1
        refine0 = torch.cat((feature_0[:, :3, :, :], raw_alpha * 256), 1)
        refine1 = self.refine_conv1(refine0)
        refine2 = self.refine_conv2(refine1)
        refine3 = self.refine_conv3(refine2)
        refine_alpha = self.refine_conv_predict(refine3)
        refine_alpha = F.sigmoid(raw_alpha + refine_alpha)

        return raw_alpha, refine_alpha
  

