# -*- coding: utf-8 -*-
import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self): # base 是获得模型的参数量
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())  # 获取所有有梯度更新的参数
        params = sum([np.prod(p.size()) for p in model_parameters]) # 获取所有可训参数的数量
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
