# -*- coding: utf-8 -*-

import os
import re

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from bert4keras.snippets import sequence_padding

from zh_nlp_demo.keras.data.dataset.base_dataset import BaseDataset


class MSRA(BaseDataset):
    def __init__(self):
        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        self.dataset_path = os.path.join(project_path, 'dataset/MSRA')
        self.dataset_name = 'MSRA'
        self.dataset_task_type = 'ner'
        self.input_len = 200
        self.categories = [
            'LOC',
            'ORG',
            'PER',
        ]

    def update_model_config(self, config):
        config['categories'] = self.categories
        return config

    def get_data(self, file_path):
        """加载数据
        单条格式：[text, [start, end, label], [start, end, label], ...]，
                  意味着text[start:end + 1]是类型为label的实体。
        """
        D, d, i = [], [''], 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    D.append(d)
                    d, i = [''], 0
                    continue

                items = line.split('\t')
                if len(items) != 2:
                    continue
                char, flag = items
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                elif flag[0] == 'I':
                    d[-1][1] = i
                i += 1
        return D

    def get_train_data(self):
        file_path = os.path.join(self.dataset_path, 'msra_train_bio.txt')
        return self.get_data(file_path)

    def get_test_data(self):
        file_path = os.path.join(self.dataset_path, 'msra_test_bio.txt')
        return self.get_data(file_path)

    def encode_bert(self, D, tokenizer):
        'bert tokenizer'
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for d in D:
            tokens = tokenizer.tokenize(d[0], maxlen=self.input_len)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))

            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = self.categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = self.categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_labels = sequence_padding(batch_labels)
        return batch_token_ids, batch_segment_ids, batch_labels

    def encode(self, D, tokenizer):
        batch_token_ids = tokenizer.encode_X([d[0] for d in D], self.input_len)
        batch_labels = []

        for token_ids, d in zip(batch_token_ids, D):
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if end >= self.input_len:
                    continue
                labels[start] = self.categories.index(label) * 2 + 1
                labels[start + 1:end + 1] = self.categories.index(label) * 2 + 2
            batch_labels.append(labels)
        return batch_token_ids, np.array(batch_labels)

    def get_train_input_data(self, tokenizer):
        if tokenizer.tokenizer_type == 'bert':
            D = self.get_train_data()
            train_D, valid_D = train_test_split(D, test_size=0.1, random_state=42)
            (train_token, train_segment, train_label) = self.encode_bert(train_D, tokenizer)
            (valid_token, valid_segment, valid_label) = self.encode_bert(valid_D, tokenizer)
            return [train_token, train_segment], train_label, [valid_token, valid_segment], valid_label
        else:
            D = self.get_train_data()
            train_D, valid_D = train_test_split(D, test_size=0.1, random_state=42)
            train_token, train_label = self.encode(train_D, tokenizer)
            valid_token, valid_label = self.encode(valid_D, tokenizer)
            return train_token, train_label, valid_token, valid_label

    def get_test_input_data(self, tokenizer):
        D = self.get_test_data()
        if tokenizer.tokenizer_type == 'bert':
            return self.encode_bert(D, tokenizer)
        else:
            return self.encode(D, tokenizer)


if __name__ == '__main__':
    ds = MSRA()
    D = ds.get_train_data()
    # D = ds.get_test_data()
    print(len(D))
    print(D[:5])
