# -*- coding: utf-8 -*-
import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import cv2
import numpy as np
import json
from model.loss import alpha_prediction_loss, compositional_loss, overall_loss
import torch.utils.data as data
import torch.utils.data.sampler as sampler
from utils import get_final_output as get_final_output

class alpha_com_test(data.Dataset):
    def __init__(self, config):
        self.image_paths = config['data_loader']['alpha_com_input']
        self.trimap1_paths = config['data_loader']['alpha_com_trimap1']
        self.trimap2_paths = config['data_loader']['alpha_com_trimap2']
        self.trimap3_paths = config['data_loader']['alpha_com_trimap3']

        self.image_files = module_data.get_files(self.image_paths)
        self.samples = []
        for image_name in self.image_files:
            image_path = os.path.join(self.image_paths, image_name)
            trimap1_path = os.path.join(self.trimap1_paths, image_name)
            trimap2_path = os.path.join(self.trimap2_paths, image_name)
            trimap3_path = os.path.join(self.trimap3_paths, image_name)
            self.samples.append((image_path, trimap1_path, trimap2_path, trimap3_path, image_name))

    def __getitem__(self, item):
        image_path, trimap1_path, trimap2_path, trimap3_path, image_name = self.samples[item]
        image = cv2.imread(image_path)
        trimap1 = cv2.imread(trimap1_path)
        trimap1 = cv2.cvtColor(trimap1, cv2.COLOR_RGB2GRAY)
        trimap2 = cv2.imread(trimap2_path)
        trimap2 = cv2.cvtColor(trimap2, cv2.COLOR_RGB2GRAY)
        trimap3 = cv2.imread(trimap3_path)
        trimap3 = cv2.cvtColor(trimap3, cv2.COLOR_RGB2GRAY)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        trimap1 = torch.from_numpy(trimap1.astype(np.float32)[np.newaxis, :, :])
        trimap2 = torch.from_numpy(trimap2.astype(np.float32)[np.newaxis, :, :])
        trimap3 = torch.from_numpy(trimap3.astype(np.float32)[np.newaxis, :, :])

        return image, trimap1, trimap2, trimap3, image_name

    def __len__(self):
        return len(self.samples)


def alpha_com_dataloader(config, ):
    dataset = alpha_com_test(config)
    test_data_loader = data.DataLoader(dataset=dataset, batch_size=1,
                                       num_workers=0)

    return test_data_loader


def test_alphacom(config, resume):
    config['data_loader']['args']['batch_size'] = 1
    test_data_loader = alpha_com_dataloader(config)
    batch_size = config['data_loader']['args']['batch_size']

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 一般有drop 或者 bn用这个

    with torch.no_grad():
        # for i, data in enumerate(tqdm(test_data_loader)):
        for i, data in enumerate(test_data_loader):
            image = data[0][0].permute(1, 2, 0)
            h, w, c = image.size()
            new_h = h - h % 32
            new_w = w - w % 32
            # print(image.size(), h, w)
            image = image.numpy()
            # print(image.shape, new_h, new_w)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # print(image.shape)
            image = torch.from_numpy(image).permute(2, 0, 1)[np.newaxis, :, :, :].to(device)
            trimap1 = data[1][0].permute(1, 2, 0)
            trimap2 = data[2][0].permute(1, 2, 0)
            trimap3 = data[3][0].permute(1, 2, 0)
            image_name = data[4][0]

            # print(trimap1.size())
            trimap1 = trimap1.numpy()
            # print(trimap1.shape)
            # for i in trimap1: print(i)
            # cv2.imshow("213414",trimap1)
            # cv2.waitKey(0)
            trimap1 = cv2.resize(trimap1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            trimap1 = torch.from_numpy(trimap1)[np.newaxis, np.newaxis, :, :].to(device)
            trimap2 = trimap2.numpy()
            trimap2 = cv2.resize(trimap2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            trimap2 = torch.from_numpy(trimap2)[np.newaxis, np.newaxis, :, :].to(device)
            trimap3 = trimap3.numpy()
            trimap3 = cv2.resize(trimap3, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            trimap3 = torch.from_numpy(trimap3)[np.newaxis, np.newaxis, :, :].to(device)

            # print(image.size(), trimap1.size(), trimap2.size(), trimap3.size())
            if config['arch']['args']['stage'] == 0:
                pred1 = model(torch.cat((image, trimap1), dim=1))
                pred2 = model(torch.cat((image, trimap2), dim=1))
                pred3 = model(torch.cat((image, trimap3), dim=1))
            else:
                _, pred1 = model(torch.cat((image, trimap1), dim=1))
                _, pred2 = model(torch.cat((image, trimap2), dim=1))
                _, pred3 = model(torch.cat((image, trimap3), dim=1))

            pred1 *= 255
            pred2 *= 255
            pred3 *= 255
            pred1 = pred1[0].permute(1, 2, 0).cpu().numpy()
            image1 =  get_final_output(pred1, trimap1)
            image1 = cv2.resize(image1, (w, h), interpolation=cv2.INTER_LINEAR)
            pred2 = pred2[0].permute(1, 2, 0).cpu().numpy()
            image2 =  get_final_output(pred1, trimap2)
            image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_LINEAR)
            pred3 = pred3[0].permute(1, 2, 0).cpu().numpy()
            image3 =  get_final_output(pred1, trimap3)
            image3 = cv2.resize(image3, (w, h), interpolation=cv2.INTER_LINEAR)

            # print(image.size())
            # print(pred1.shape,pred2.shape,pred3.shape)
            # pred3 = cv2.resize(pred3, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("alpha_com_output/Trimap1/{}".format(image_name), image1)
            cv2.imwrite("alpha_com_output/Trimap2/{}".format(image_name), image2)
            cv2.imwrite("alpha_com_output/Trimap3/{}".format(image_name), image3)
            print(i)


def DIM_test(config, resume):
    # setup data_loader instances
    config['data_loader']['args']['batch_size'] = 1
    config['data_loader']['args']['usage'] = 'test'
    config['data_loader']['args']['validation_split'] = 0
    test_data_loader = get_instance(module_data, 'data_loader', config, config)
    batch_size = config['data_loader']['args']['batch_size']

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model.eval()  # 一般有drop 或者 bn用这个

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    cot = 0
    total_SAD = 0.0
    total_MSE = 0.0
    total_loss = 0.0

    with torch.no_grad():
        # for i, data in enumerate(tqdm(test_data_loader)):
        for i, data in enumerate(test_data_loader):
            image = data[0].to(device)
            alpha = data[1].to(device)
            fg = data[2].to(device)
            bg = data[3].to(device)
            trimap = data[4].to(device)
            if config['arch']['args']['stage'] == 0:
                raw_alpha_pred = model(torch.cat((image, trimap), dim=1))
            else:
                raw_alpha_pred, refine_alpha_pred = model(torch.cat((image, trimap), dim=1))
            #
            # save sample images, or do something with output here
            #
            #
            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            w = 0.5  # 0.25 0.75
            loss = w * overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg) + \
                   (1 - w) * alpha_prediction_loss(alpha, refine_alpha_pred, trimap)

            metric_s = ''
            acc_metrics = np.zeros(len(metric_fns))  # 清空list
            for j, metric in enumerate(metric_fns):  # 对所有的metrics进行评测
                acc_metrics[j] += metric(refine_alpha_pred, alpha, trimap)
                metric_s += metric.__name__ + ':  ' + str(acc_metrics[j]) + '   '
            total_MSE += acc_metrics[0]
            total_SAD += acc_metrics[1]

            refine_alpha_pred *= 255
            for j in range(batch_size):
                cv2.imwrite("test_output/{}original.jpg".format(cot), image[j].permute(1, 2, 0).cpu().numpy())
                cv2.imwrite("test_output/{}alpha_matte.jpg".format(cot),
                            refine_alpha_pred[j].permute(1, 2, 0).cpu().numpy())
                cot += 1

            total_loss += loss.item() * batch_size  # loss是取平均的
            print("test{}/{}:  loss: {}  ".format(i, len(test_data_loader), loss / (batch_size), ))
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size
    print("avg_LOSS:  {}   avg_SAD: {},   avg_MSE:  {}".format(total_loss / len(test_data_loader),
                                                               total_SAD / len(test_data_loader),
                                                               total_MSE / len(test_data_loader)))

    # n_samples = len(test_data_loader.sampler)
    # log = {'loss': total_loss / n_samples}  # 整个echo的loss
    # log.update(
    #     {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})  # 不同评测方式的结果,并且更新值
    # print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-a', '--alpha', default=0, type=int,
                        help='indices of use alpha.com or not(default: 0)')

    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    elif args.resume:
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.alpha == 0:
        DIM_test(config, args.resume)
    else:
        test_alphacom(config, args.resume)
