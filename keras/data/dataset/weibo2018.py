# -*- coding: utf-8 -*-

import os
import re

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from zh_nlp_demo.keras.data.dataset.base_dataset import BaseDataset


class WeiBo2018(BaseDataset):
    def __init__(self):
        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        self.dataset_path = os.path.join(project_path, 'dataset/weibo2018')
        self.dataset_name = 'weibo2018'
        self.dataset_task_type = 'classification'
        self.class_num = 2
        self.input_len = 200

    def update_model_config(self, config):
        config['class_num'] = self.class_num
        return config

    def process(self, text: str) -> str:
        """数据处理，清洗"""

        # 去除 {%xxx%} (地理定位, 微博话题等)
        text = re.sub('\{%.+?%\}', ' ', text)
        # 去除 @xxx (用户名)
        text = re.sub('@.+?( |$)', ' ', text)
        # 去除 【xx】 (里面的内容通常都不是用户自己写的)
        text = re.sub('【.+?】', ' ', text)
        # '\u200b'是这个数据集中的一个bad case,
        text = re.sub('\u200b', ' ', text)
        return text

    def get_data(self, file_path):
        X, Y = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                [_, seniment, content] = line.split(',', 2)
                content = self.process(content.strip())
                X.append(content)
                Y.append(int(seniment))
        return X, Y

    def get_train_data(self):
        file_path = os.path.join(self.dataset_path, 'train.txt')
        return self.get_data(file_path)

    def get_test_data(self):
        file_path = os.path.join(self.dataset_path, 'test.txt')
        return self.get_data(file_path)

    def get_train_input_data(self, tokenizer):
        raw_x, raw_y = self.get_train_data()
        tokens = [tokenizer.encode(text) for text in raw_x]
        X = pad_sequences(tokens, maxlen=self.input_len)
        Y = to_categorical(raw_y, self.class_num)
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.1, random_state=42)
        return train_X, train_Y, valid_X, valid_Y

    def get_test_input_data(self, tokenizer):
        raw_x, raw_y = self.get_test_data()
        tokens = [tokenizer.encode(text) for text in raw_x]
        X = pad_sequences(tokens, maxlen=self.input_len)
        Y = to_categorical(raw_y, self.class_num)
        return (X, Y)


if __name__ == '__main__':
    ds = WeiBo2018()
    x, y = ds.get_train_data()
    print(len(x), len(y))
    print(x[0])
    print(y[0])
