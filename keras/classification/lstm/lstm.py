# -*- coding: utf-8 -*-

import os
import abc
import argparse
import warnings

import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard

from zh_nlp_demo.keras.data.dataset.weibo2018 import WeiBo2018
from zh_nlp_demo.keras.data.tokenizers.char_tokenizer import CharTokenizer


warnings.filterwarnings("ignore")
tf.compat.v1.set_random_seed(42)


class DLTrainer(metaclass=abc.ABCMeta):
    def __init__(self, model_name='base'):
        self.model_name = model_name

        # 模型相关参数
        self.batch_size = 256
        self.epochs = 10
        self.verbose = 1
        self.validation_split = 0.1

        # 相关文件路径设置
        self.base_output_dir = './outputs'
        self.checkpoint_dir = os.path.join(self.base_output_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_dir = os.path.join(self.base_output_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 组件
        self.dataset = WeiBo2018()
        token_dict = self._load_dict()
        self.tokenizer = CharTokenizer(token_dict)

    def _load_dict(self):
        " 加载词表字典"
        #  TODO: 考虑把字典单独抽象出来 #

        token_dict = {}
        project_path = os.environ.get('ZH_NLP_DEMO_PATH')
        Vocabulary_file = os.path.join(project_path, 'keras/data/dict/char_vocab.dic')
        with open(Vocabulary_file, 'r', encoding='utf-8') as f:
            for line in f:
                items = line.strip().split('\t')
                token_dict[items[0]] = int(items[1])
        return token_dict

    @abc.abstractmethod
    def make_model(self):
        pass

    def train(self):
        train_x, train_y = self.dataset.get_train_input_data(self.tokenizer)

        checkpoint_file = os.path.join(self.checkpoint_dir, f'{self.model_name}_' + '{epoch:03d}_acc{val_accuracy:.2f}.hdf5')
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_accuracy', verbose=1, save_best_only=True)

        tensorboard = TensorBoard(log_dir=self.log_dir)

        model = self.make_model()
        model.fit(
            train_x, train_y, batch_size=256, epochs=20, verbose=1, validation_split=0.1, callbacks=[checkpoint, tensorboard]
        )


class NativeLSTMTrainer(DLTrainer):

    def __init__(self):
        super(NativeLSTMTrainer, self).__init__('native_lstm')

    def make_model(self):
        model = Sequential()
        model.add(Embedding(7000, 64))
        model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


class BiLSTMTrainer(DLTrainer):

    def __init__(self):
        super(BiLSTMTrainer, self).__init__('bi_lstm')

    def make_model(self):
        model = Sequential()
        model.add(Embedding(7000, 64))
        model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于lstm的文本分类')
    model_type = ('native_lstm', 'bi_lstm')
    parser.add_argument(
        '-m', '--model_type', type=str, default='native_lstm',
        choices=model_type, help='指定模型架构'
    )
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    args = parser.parse_args()

    model_type = args.model_type
    if model_type == 'bi_lstm':
        trainer = BiLSTMTrainer()
    else:
        trainer = NativeLSTMTrainer()

    if args.do_train:
        trainer.train()

    if args.do_test:
        pass
