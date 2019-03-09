# -*- coding: utf-8 -*-
import os
import importlib
import warnings


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable: # 如果打开图示功能
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path) # 创建一个 名为log_path的存放画图的文件
            except ModuleNotFoundError:
                message = """TensorboardX visualization is configured to use, but currently not installed on this machine. Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."""
                warnings.warn(message, UserWarning)
                logger.warn(message) # 这里不加里面的message会报错 可能是版本问题
        self.step = 0
        self.mode = ''
        # 以下是几个tensorboard提供的可视化方式
        self.tensorboard_writer_ftns = ['add_graph', 'add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):  # 设置每次画图的step
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns: # 如果name存在
            add_data = getattr(self.writer, name, None) # 获得writer里可视化方法
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs) # 传入参数到方法里
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr
