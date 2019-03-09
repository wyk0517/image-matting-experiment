# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger


def get_instance(module, name, config, *args): # 从模型中获取config配置的对象， name是config文件的分类， type是具体的哪一个网络
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()  # 训练日志

    # setup data_loader instances
    if config['data_loader']['type']=='dataloader_test':
        train_data_loader = get_instance(module_data, 'data_loader', config, config)
        config['data_loader']['args']['usage'] = "test"
        config['data_loader']['args']['transform_switch'] = False
        #config['data_loader']['args']['batch_size'] = 1
        valid_data_loader = get_instance(module_data, 'data_loader', config, config)
    else:
        train_data_loader,valid_data_loader = get_instance(module_data, 'data_loader', config, config)
    # valid_data_loader = data_loader.split_validation()  # 获取验证集

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()  # base是 获取可训参数的数量

    # get function handles of loss and metrics
    # loss = getattr(module_loss, config['loss'])  # 获取loss
    loss = [getattr(module_loss, los) for los in config['loss']]  # 获取loss
    metrics = [getattr(module_metric, met) for met in config['metrics']]  # 获取metric

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())  # 获取可训参数
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)  # 优化器
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer) # 获得lr配置
    trainer = Trainer(model, loss, metrics, optimizer,  # 构建训练器
                      resume=resume,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()  # 获取训练日志

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])  # 合成新的路径
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)
