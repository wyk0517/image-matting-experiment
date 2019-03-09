# -*- coding: utf-8 -*-
import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX
from collections import OrderedDict


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__) # 生成一个对应类的日志

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1: # 如果gpu > 1并行gpu
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        #self.device = torch.device('cuda:0' if config['n_gpu'] > 0 else 'cpu')
        #self.model = model.to(self.device)
        #if config['n_gpu'] > 1:
            #self.model = torch.nn.DataParallel(model)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.epochs = config['trainer']['epochs']
        self.save_freq = config['trainer']['save_freq']
        self.verbosity = config['trainer']['verbosity']

        self.train_logger = train_logger

        # configuration to monitor model performance and save best
        self.monitor = config['trainer']['monitor']
        self.monitor_mode = config['trainer']['monitor_mode']
        assert self.monitor_mode in ['min', 'max', 'off'] # 判断monitor_mode是否是这三个 不是抛出异常
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf # 初始化模型最优loss
        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S') # 获取当前时间
        self.checkpoint_dir = os.path.join(config['trainer']['save_dir'], config['name'], start_time) # checkpoint的目录（名字）
        # setup visualization writer instance
        writer_dir = os.path.join(config['visualization']['log_dir'], config['name'], start_time) # 可视化的地址
        self.writer = WriterTensorboardX(writer_dir, self.logger, config['visualization']['tensorboardX']) # 第三个参数决定是否开启可视化

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)  # 如果checkpoint_dir不存在创建一个
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')  # path
        with open(config_save_path, 'w') as handle:  # 把路径存到config
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
    
    def _prepare_device(self, n_gpu_use):
        """ 
        setup GPU device if available, move model into configured device
        """ 
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            self.logger.warning(msg)
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self): # 训练的日志
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            
            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off': # 获取best的参数
                try:
                    if  (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                        (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                        self.monitor_best = log[self.monitor]
                        best = True
                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
                            + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.logger.warning(msg)
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _train_epoch(self, epoch): # 训练一次
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
        #                         'This may yield an exception while state_dict is being loaded.')

        ########################
        # load a part of the net
        #########################
        pretrained_dict = checkpoint['state_dict']
        model_dict = self.model.state_dict()
        if checkpoint['config']['n_gpu'] > 1 and self.config['n_gpu'] == 1:
            new_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]
                new_dict[name] = v
            pretrained_dict = new_dict
        elif checkpoint['config']['n_gpu'] == 1 and self.config['n_gpu'] > 1:
            new_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = "module."+k
                new_dict[name] = v
            pretrained_dict = new_dict
        print("The pretrained model's para is following")
        for k, v in pretrained_dict.items():
            print(k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # load optimizer state from checkpoint only when optimizer type is not changed. 
        #####################################################
        # when fine-tuning, don't load pre-optim
        ######################################################
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            pass
    
        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
