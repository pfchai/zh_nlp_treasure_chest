# -*- coding: utf-8 -*-

import os
import re
import json

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class CAIL2018():
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

        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        if sub_data == 'exercise':
            self.dataset_path = os.path.join(project_path, 'dataset/CAIL2018/final_all_data/exercise_contest')
            self.train_data = os.path.join(self.dataset_path, 'data_train.json')
            self.test_data = os.path.join(self.dataset_path, 'data_train.json')
        elif sub_data == 'first_stage':
            self.dataset_path = os.path.join(project_path, 'dataset/CAIL2018/final_all_data/first_stage')
            self.train_data = os.path.join(self.dataset_path, 'train.json')
            self.test_data = os.path.join(self.dataset_path, 'train.json')
        else:
            raise '无效数据集'

        # 语料库相关配置
        if task_name == 'accusation':
            self.label_count = 202
        elif task_name == 'relevant_articles':
            self.label_count = 183
        elif task_name == 'term_of_imprisonment':
            # 0-25年 单位为月 25*12=300; 无期: 301; 死刑: 302
            self.label_count = 303
        else:
            raise '无效任务名称'

        # 标签编号字典
        self.accu_dict = {}
        accu_dict_path = os.path.join(project_path, 'dataset/CAIL2018/final_all_data/accu.txt')
        with open(accu_dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                self.accu_dict[line.strip()] = index

        self.law_dict = {}
        law_dict_path = os.path.join(project_path, 'dataset/CAIL2018/final_all_data/law.txt')
        with open(law_dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                self.law_dict[int(line.strip())] = index

    def update_model_config(self, config):
        config['class_num'] = self.label_count
        if self.task_name in ('accusation', 'relevant_articles'):
            config['activation'] = 'sigmoid'
            config['compile']['loss'] = 'binary_crossentropy'
        return config

    def get_train_data(self):
        train_x, train_y = [], []
        with open(self.train_data, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                train_x.append(data['fact'])
                train_y.append(data['meta'][self.task_name])
        return train_x, train_y

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

    def get_train_input_data(self, tokenizer):
        raw_X, raw_Y = self.get_train_data()
        tokens = [tokenizer.encode(text) for text in raw_X]
        X = pad_sequences(tokens, maxlen=500)
        Y = self.encode_Y(raw_Y)
        return (X, Y)


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
