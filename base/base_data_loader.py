# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)  # 样本数量

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)  # 获取训练样本与测试样本

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)  # 传入dataloader

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)  # 生成一个1-n的list

        np.random.seed(0)  # 随机种子(numpy)
        np.random.shuffle(idx_full)  # 随机

        len_valid = int(self.n_samples * split)  # 验证数据集大小

        valid_idx = idx_full[0:len_valid] # 获取验证样本
        train_idx = np.delete(idx_full, np.arange(0, len_valid)) # 把验证集的序号删掉
        
        train_sampler = SubsetRandomSampler(train_idx)  # 获取训练样本
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)  # 样本数改为训练数据集设数量

        return train_sampler, valid_sampler
        
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

