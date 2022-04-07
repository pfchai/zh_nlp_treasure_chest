# -*- coding: utf-8 -*-

import os
from collections import Counter


class Vocabulary():
    def __init__(self):
        self.token_dict = {}
        self.token_dict_inv = {}

    def check_created(func):
        def _check_created(self, *args, **kwargs):
            if self.token_dict or self.token_dict_inv:
                raise '已经创建过词典'
            return func(self, *args, **kwargs)
        return _check_created

    @check_created
    def create_from_file(self, file_path=None):
        " 加载词表字典"

        if file_path is None:
            project_path = os.environ.get('ZH_NLP_DEMO_PATH')
            vocabulary_file = os.path.join(project_path, 'keras/data/dict/char_vocab.dic')
        else:
            vocabulary_file = file_path
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.strip().split('\t')
                self.token_dict[items[0]] = int(items[1])
                self.token_dict_inv[int(items[1])] = items[0]

    @check_created
    def create_from_counter(self, counter, min_count=0):
        # word_counts = sorted(dict(counter).items(), key=lambda x: x[1], reverse=True)
        vocab_words = ["<PAD>", "<UNK>"]
        for w, c in counter.most_common(100):
            if c < min_count:
                break
            vocab_words.append(w)

        self.token_dict = {w: i for i, w in enumerate(vocab_words)}
        self.token_dict_inv = {i: w for i, w in enumerate(vocab_words)}

    @check_created
    def create_from_corpus(self, corpus, is_word_split=False, min_count=0):
        counter = Counter()
        for text in corpus:
            if is_word_split:
                raise '未实现'
            else:
                counter.update(text)
        self.create_from_counter(counter, min_count)

    def save_to_files(self, path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            for i in range(len(self.token_dict_inv)):
                f.write('{}\t{}\n'.format(i, self.token_dict_inv[i]))


if __name__ == '__main__':
    vocabulary = Vocabulary()
    vocabulary.create_from_file()
    vocabulary.create_from_corpus(['你好'])
    print(len(vocabulary.token_dict))

