# -*- coding: utf-8 -*-

import os
import warnings

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField

from zh_nlp_demo.keras.ner.utils import create_parser, create_dataset
from zh_nlp_demo.keras.trainer import Trainer
from zh_nlp_demo.keras.data.vocabulary import Vocabulary
from zh_nlp_demo.keras.data.tokenizers.char_tokenizer import CharTokenizer


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
tf.compat.v1.set_random_seed(42)


# 默认参数配置
default_config = {
    'vocab_size': 7000,
    'emb_hidden_size': 64,
    'categories': [],
    'crf_lr_multiplier': 1000,
    'compile': {
        'learning_rate': 2e-5,
    },
    'train_config': {
        'batch_size': 256,
        'epochs': 10,
        'verbose': 1,
    },
}


def make_model(config):
    categories = config['categories']
    crf_lr_multiplier = config['crf_lr_multiplier']

    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    model = Sequential()
    model.add(Embedding(config['vocab_size'], 64))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Dense(len(categories) * 2 - 1))
    model.add(CRF)

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(config['compile']['learning_rate']),
        metrics=[CRF.sparse_accuracy]
    )
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于BiLSTM+CRF的命名实体识别')
    args = parser.parse_args()

    config = default_config

    # 数据集
    dataset = create_dataset(args)
    config = dataset.update_model_config(config)
    print('dataset load finished')

    vocabulary = Vocabulary()
    vocabulary.create_from_file()
    tokenizer = CharTokenizer(vocabulary=vocabulary)
    config['vocab_size'] = tokenizer.vocab_size

    model = make_model(config)
    model.summary()

    trainer = Trainer(model_name='bi_lstm_crf', dataset=dataset, tokenizer=tokenizer, model=model)
    trainer.monitor = 'sparse_accuracy'

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
