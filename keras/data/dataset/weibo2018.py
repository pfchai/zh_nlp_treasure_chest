# -*- coding: utf-8 -*-

import os
import re

from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class WeiBo2018():
    def __init__(self):
        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        self.dataset_path = os.path.join(project_path, 'dataset/weibo2018')
        self.dataset_name = 'weibo2018'
        self.dataset_task_type = 'classification'

    def update_model_config(self, config):
        config['class_num'] = 2
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

    def get_train_data(self):
        train_x, train_y = [], []
        file_path = os.path.join(self.dataset_path, 'train.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                [_, seniment, content] = line.split(',', 2)
                content = self.process(content.strip())
                train_x.append(content)
                train_y.append(int(seniment))
        return train_x, train_y

    def get_train_input_data(self, tokenizer):
        raw_x, raw_y = self.get_train_data()
        tokens = [tokenizer.encode(text) for text in raw_x]
        x = pad_sequences(tokens, maxlen=100)
        y = to_categorical(raw_y, 2)
        return (x, y)


if __name__ == '__main__':
    ds = WeiBo2018()
    x, y = ds.get_train_data()
    print(len(x), len(y))
    print(x[0])
    print(y[0])
