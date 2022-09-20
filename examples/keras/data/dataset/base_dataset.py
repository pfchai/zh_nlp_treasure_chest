# -*- coding: utf-8 -*-

import os
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    def __init__(self):
        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        self.dataset_base = os.path.join(project_path, 'dataset')

    @abc.abstractmethod
    def update_model_config(self, config):
        "根据数据集，更新模型相关参数设置"
        pass

    @abc.abstractmethod
    def get_train_data(self):
        "获取原始训练数据"
        pass

    @abc.abstractmethod
    def get_test_data(self):
        "获取原始测试数据"
        pass

    @abc.abstractmethod
    def get_train_input_data(self, tokenizer):
        "获取模型输入的训练数据"
        pass

    @abc.abstractmethod
    def get_test_input_data(self, tokenizer):
        "获取模型输入的测试数据"
        pass
