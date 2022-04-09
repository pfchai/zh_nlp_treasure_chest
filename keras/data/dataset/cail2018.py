# -*- coding: utf-8 -*-

import os
import re
import json

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from zh_nlp_demo.keras.data.dataset.base_dataset import BaseDataset


class CAIL2018(BaseDataset):
    def __init__(self, task_name='accusation', sub_data='exercise'):
        """
        task_name: 任务名称
            accusation - 任务一（罪名预测）：根据刑事法律文书中的案情描述和事实部分，预测被告人被判的罪名；
            relevant_articles - 任务二（法条推荐）：根据刑事法律文书中的案情描述和事实部分，预测本案涉及的相关法条；
            term_of_imprisonment - 任务三（刑期预测）：根据刑事法律文书中的案情描述和事实部分，预测被告人的刑期长短。
        sub_data: 使用哪个数据集
            exercise - 练习赛
            first - 第一阶段正赛
        """

        self.task_name = task_name
        self.sub_data = sub_data
        self.dataset_name = 'CAIL2018'
        self.dataset_task_type = 'classification'
        self.input_len = 500

        self.project_path = os.environ.get('ZH_NLP_DEMO_PATH')

        # 语料库相关配置
        if self.task_name == 'accusation':
            self.label_count = 202
        elif self.task_name == 'relevant_articles':
            self.label_count = 183
        elif self.task_name == 'term_of_imprisonment':
            # 0-25年 单位为月 25*12=300; 无期: 301; 死刑: 302
            self.label_count = 303
        else:
            raise '无效任务名称'

        # 标签编号字典
        self.accu_dict = {}
        accu_dict_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/accu.txt')
        with open(accu_dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                self.accu_dict[line.strip()] = index

        self.law_dict = {}
        law_dict_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/law.txt')
        with open(law_dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                self.law_dict[int(line.strip())] = index

    def update_model_config(self, config):
        config['class_num'] = self.label_count
        if self.task_name in ('accusation', 'relevant_articles'):
            config['activation'] = 'sigmoid'
            config['compile']['loss'] = 'binary_crossentropy'
        return config

    def encode_Y(self, raw_Y):
        Y = []
        if self.task_name == 'accusation':
            for raw_y in raw_Y:
                y = [0] * self.label_count
                for i in raw_y:
                    i = re.sub(r'\[|\]', '', i)
                    y[self.accu_dict[i]] = 1.
                Y.append(y)
            Y = np.array(Y)

        if self.task_name == 'relevant_articles':
            for raw_y in raw_Y:
                y = [0] * self.label_count
                for i in raw_y:
                    i = int(i)
                    y[self.law_dict[i]] = 1.
                Y.append(y)
            Y = np.array(Y)

        if self.task_name == 'term_of_imprisonment':
            for raw_y in raw_Y:
                if raw_y['life_imprisonment']:
                    y = 301
                elif raw_y['death_penalty']:
                    y = 302
                else:
                    y = raw_y['imprisonment']
                Y.append(y)
            Y = to_categorical(Y, self.label_count)
        return Y

    def get_data(self, file_name):
        train_x, train_y = [], []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                train_x.append(data['fact'])
                train_y.append(data['meta'][self.task_name])
        return train_x, train_y

    def get_train_data(self):
        if self.sub_data == 'exercise':
            dataset_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/exercise_contest')
            train_file_path = os.path.join(dataset_path, 'data_train.json')
            valid_file_path = os.path.join(dataset_path, 'data_valid.json')
            train_x, train_y = self.get_data(train_file_path)
            valid_x, valid_y = self.get_data(valid_file_path)

        if self.sub_data == 'first_stage':
            dataset_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/first_stage')
            train_file_path = os.path.join(self.dataset_path, 'train.json')
            train_x, train_y = self.get_data(train_file_path)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

        return train_x, train_y, valid_x, valid_y

    def get_test_data(self):
        if self.sub_data == 'exercise':
            dataset_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/exercise_contest')
            file_path = os.path.join(dataset_path, 'data_test.json')
            X, Y = self.get_data(file_path)

        if self.sub_data == 'first_stage':
            dataset_path = os.path.join(self.project_path, 'dataset/CAIL2018/final_all_data/first_stage')
            file_path = os.path.join(self.dataset_path, 'test.json')
            X, Y = self.get_data(file_path)

        return X, Y

    def get_train_input_data(self, tokenizer):
        train_X, train_Y, valid_X, valid_Y = self.get_train_data()
        train_X = tokenizer.encode_X(train_X, self.input_len)
        valid_X = tokenizer.encode_X(valid_X, self.input_len)
        train_Y = self.encode_Y(train_Y)
        valid_Y = self.encode_Y(valid_Y)
        return train_X, train_Y, valid_X, valid_Y

    def get_test_input_data(self, tokenizer):
        test_X, test_Y = self.get_test_data()
        tokens = [tokenizer.encode(text) for text in test_X]
        X = pad_sequences(tokens, maxlen=self.input_len)
        Y = self.encode_Y(test_Y)
        return X, Y


if __name__ == '__main__':
    for task_name in ('accusation', 'relevant_articles', 'term_of_imprisonment'):
        for sub_data in ('exercise', 'first_stage'):
            ds = CAIL2018(task_name=task_name, sub_data=sub_data)
            x, y = ds.get_train_data()
            print('--' * 50)
            print(task_name, sub_data)
            print(len(x), len(y))
            print(x[0])
            print(y[0])
