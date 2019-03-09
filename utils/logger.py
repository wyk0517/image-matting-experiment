# -*- coding: utf-8 -*-
import json
import logging

logging.basicConfig(level=logging.INFO, format='')

class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry # 增加日志

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)  # 将 Python 对象编码成 JSON 字符串
