# -*- coding: utf-8 -*-

import argparse
import warnings

import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, Dropout

from zh_nlp_demo.keras.trainer import Trainer
from zh_nlp_demo.keras.data.dataset.weibo2018 import WeiBo2018
from zh_nlp_demo.keras.data.vocabulary import Vocabulary
from zh_nlp_demo.keras.data.tokenizers.char_tokenizer import CharTokenizer


warnings.filterwarnings("ignore")
tf.compat.v1.set_random_seed(42)


# 默认参数配置
default_config = {
    'vocab_size': 7000,
    'emb_hidden_size': 64,
    'class_num': 2,
    'train_config': {
        'batch_size': 256,
        'epochs': 10,
        'validation_split': 0.1,
        'verbose': 1,
    }
}


def make_model(config):
    model = Sequential()
    model.add(Embedding(config['vocab_size'], 64))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(config['class_num'], activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于LSTM的文本分类')
    parser.add_argument(
        '--dataset', type=str, default='weibo2018',
        choices=('weibo2018',), help='指定数据集'
    )
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    args = parser.parse_args()

    config = default_config

    # 数据集
    if args.dataset == 'weibo2018':
        config['class_num'] = 2
        dataset = WeiBo2018()
        config.update(dataset.model_config)
    else:
        raise '不支持的数据集'

    vocabulary = Vocabulary()
    vocabulary.create_from_file()
    tokenizer = CharTokenizer(vocabulary=vocabulary)
    config['vocab_size'] = tokenizer.vocab_size

    model = make_model(config)

    trainer = Trainer(model_name='lstm', dataset=dataset, tokenizer=tokenizer, model=model)

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
