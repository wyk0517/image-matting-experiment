# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from base import BaseTrainer
from model.loss import alpha_prediction_loss, compositional_loss, overall_loss,OHEM_loss, point_OHEM_loss


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = data_loader.batch_size

    def _eval_metrics(self, pred, alpha, trimap):
        acc_metrics = np.zeros(len(self.metrics))  # 清空list
        for i, metric in enumerate(self.metrics):  # 对所有的metrics进行评测
            acc_metrics[i] += metric(pred, alpha, trimap)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])  # 记录到writer里
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        radio = self.config['ohem_loss']
        point_ohem_loss = OHEM_loss(self.config['arch']['args']['stage'], radio=radio)
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        train_num = len(self.data_loader.dataset)

        #   print(train_num)
        for batch_idx, data in enumerate(self.data_loader):  # 加载一个batch_size的数据集
            """
            img = Variable(data[0])
            alpha = Variable(data[1])
            fg = Variable(data[2])
            bg = Variable(data[3])
            trimap = Variable(data[4])
            """
            image = data[0].to(self.device)
            alpha = data[1].to(self.device)
            fg = data[2].to(self.device)
            bg = data[3].to(self.device)
            trimap = data[4].to(self.device)
            # img_info = data[5].to(self.device)

            # print("image: {} alpha: {} fg: {} bg:{} trimap:{}".format(image.shape, alpha.shape, fg.shape, bg.shape,
                                                                      # trimap.shape))
            # input()
            self.optimizer.zero_grad()  # 每次清空梯度

            if self.config['arch']['type'] == 'rcf_vgg':
                if self.config['arch']['args']['stage'] == 0:
                    stage_pred = self.model(torch.cat((image, trimap), dim=1))
                    raw_alpha_pred = stage_pred[-1]
                else:
                    stage_pred = self.model(torch.cat((image, trimap), dim=1))
                    raw_alpha_pred, refine_alpha_pred = stage_pred[-2:]
            else:
                if self.config['arch']['args']['stage'] == 0:
                    raw_alpha_pred = self.model(torch.cat((image, trimap), dim=1))  # [batch_size, 4, H, W]
                else:
                    raw_alpha_pred, refine_alpha_pred = self.model(torch.cat((image, trimap), dim=1))
            # self.writer.add_graph(self.model, (torch.cat((image, trimap), dim=1),))
            # print(raw_alpha_pred.shape)

            #if self.config['arch']['args']['stage'] == 0:
            #    loss = overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg)
            #elif self.config['arch']['args']['stage'] == 1:
            #    loss = alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
            #elif self.config['arch']['args']['stage'] == 2:
            #    w = 0 # 0.25 0.75
            #    loss = w * overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg) + \
            #           (1-w)*alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
            #else:
            #    assert()
            ##########################################
            #
            # OHEM loss
            #
            #########################################
            stage_loss = []
            if self.config['arch']['type'] == 'rcf_vgg':
                if self.config['arch']['args']['stage'] == 0:
                    loss = torch.tensor(0).float().cuda()
                    for pred in stage_pred:
                        val = point_ohem_loss(image=image, alpha=alpha, raw_alpha_pred=pred, trimap=trimap, fg=fg, bg=bg)
                        loss = loss + val
                        stage_loss.append(val)
                elif self.config['arch']['args']['stage'] == 1:
                    loss = point_ohem_loss(alpha=alpha, refine_alpha_pred=refine_alpha_pred, trimap=trimap)
                elif self.config['arch']['args']['stage'] == 2:
                    loss = point_ohem_loss(alpha=alpha, refine_alpha_pred=refine_alpha_pred, trimap=trimap)
                    for pred in stage_pred:
                        val = point_ohem_loss(image=image, alpha=alpha, raw_alpha_pred=pred, trimap=trimap, fg=fg, bg=bg)
                        loss += 0.1*val
                        stage_loss.append(val)
            else:
                if self.config['arch']['args']['stage'] == 0:
                    loss = point_ohem_loss(image=image, alpha=alpha, raw_alpha_pred=raw_alpha_pred, trimap=trimap, fg=fg, bg=bg)
                elif self.config['arch']['args']['stage'] == 1:
                    loss = point_ohem_loss(alpha=alpha, refine_alpha_pred=refine_alpha_pred, trimap=trimap)
                elif self.config['arch']['args']['stage'] == 2:
                    w = 0  # 0.25 0.75
                    loss = point_ohem_loss(image=image, alpha=alpha, raw_alpha_pred=raw_alpha_pred, trimap=trimap, fg=fg, bg=bg, refine_alpha_pred=refine_alpha_pred, w=w)
                else:
                    assert ()

            # print(loss.shape)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            if self.config['arch']['args']['stage'] == 0:
                total_metrics += self._eval_metrics(raw_alpha_pred, alpha, trimap)
            else:
                total_metrics += self._eval_metrics(refine_alpha_pred, alpha, trimap)
            # print(train_num)
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} lr:{}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size, # 一共处理了多少个样本
                    train_num,  # 总共的样本数
                    100.0 * batch_idx / len(self.data_loader),  # 当前进度
                    loss.item(),
                    self.optimizer.param_groups[0]['lr'])) # loss
                if self.config['arch']['type'] == 'rcf_vgg':
                    if self.config['arch']['args']['stage'] == 0 or self.config['arch']['args']['stage'] == 2:
                        for i, val in enumerate(stage_loss):
                            self.logger.info('stage{}_Loss: {:.6f}'.format(i+1, val))
                self.writer.add_image('input', make_grid(data[0].cpu(), nrow=8, normalize=True))  # make_grid把图片合在一起

        log = {
            'loss': total_loss / len(self.data_loader),  # 获得一个epoch的loss与metrics
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation: # 验证集
            val_log = self._valid_epoch(epoch) # 获取验证集的log
            log = {**log, **val_log}

        if self.lr_scheduler is not None: # 调用lr_scheduler
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        val_num = len(self.data_loader.dataset) * self.config['data_loader']['args']['validation_split']
        # print(val_num)
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                image = data[0].to(self.device)
                alpha = data[1].to(self.device)
                fg = data[2].to(self.device)
                bg = data[3].to(self.device)
                trimap = data[4].to(self.device)
                # img_info = data[5].to(self.device)

                if self.config['arch']['type'] == 'rcf_vgg':
                    if self.config['arch']['args']['stage'] == 0:
                        stage_pred = self.model(torch.cat((image, trimap), dim=1))
                        raw_alpha_pred = stage_pred[-1]
                    else:
                        stage_pred = self.model(torch.cat((image, trimap), dim=1))
                        raw_alpha_pred, refine_alpha_pred = stage_pred[-2:]
                else:
                    if self.config['arch']['args']['stage'] == 0:
                        raw_alpha_pred = self.model(torch.cat((image, trimap), dim=1))  # [batch_size, 4, H, W]
                    else:
                        raw_alpha_pred, refine_alpha_pred = self.model(torch.cat((image, trimap), dim=1))

                stage_loss = []
                if self.config['arch']['type'] == 'rcf_vgg':
                    if self.config['arch']['args']['stage'] == 0:
                        loss = torch.tensor(0).float().cuda()
                        for pred in stage_pred:
                            val = overall_loss(image, alpha, pred, trimap, fg, bg)
                            loss = loss + val
                            stage_loss.append(val)
                    elif self.config['arch']['args']['stage'] == 1:
                        loss = alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
                    elif self.config['arch']['args']['stage'] == 2:
                        w = 0
                        loss = w * overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg) + \
                               (1 - w) * alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
                        for pred in stage_pred:
                            val = overall_loss(image, alpha, pred, trimap, fg, bg)
                            loss = loss + 0.1*val
                            stage_loss.append(val)
                else:
                    if self.config['arch']['args']['stage'] == 0:
                        loss = overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg)
                    elif self.config['arch']['args']['stage'] == 1:
                        loss = alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
                    elif self.config['arch']['args']['stage'] == 2:
                        w = 0
                        loss = w*overall_loss(image, alpha, raw_alpha_pred, trimap, fg, bg) + \
                               (1-w)*alpha_prediction_loss(alpha, refine_alpha_pred, trimap)
                    else:
                        assert ()

                if self.config['arch']['type'] == 'rcf_vgg':
                    self.logger.info('Eval_Loss: {:.6f}'.format(loss))
                    if self.config['arch']['args']['stage'] == 0 or self.config['arch']['args']['stage'] == 2:
                        for i, val in enumerate(stage_loss):
                            self.logger.info('stage{}_Loss: {:.6f}'.format(i+1, val))

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                if self.config['arch']['args']['stage'] == 0:
                    total_val_metrics += self._eval_metrics(raw_alpha_pred, alpha, trimap)
                else:
                    total_val_metrics += self._eval_metrics(refine_alpha_pred, alpha, trimap)
                self.writer.add_image('input', make_grid(data[0].cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
