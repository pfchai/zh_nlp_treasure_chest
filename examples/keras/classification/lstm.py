# -*- coding: utf-8 -*-

import warnings

import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, Dropout

from zh_nlp_demo.keras.classification.utils import create_parser
from zh_nlp_demo.keras.classification.utils import create_dataset
from zh_nlp_demo.keras.trainer import Trainer
from zh_nlp_demo.keras.data.vocabulary import Vocabulary
from zh_nlp_demo.keras.data.tokenizers.char_tokenizer import CharTokenizer


warnings.filterwarnings("ignore")
tf.compat.v1.set_random_seed(42)


# 默认参数配置
default_config = {
    'vocab_size': 7000,
    'emb_hidden_size': 64,
    'class_num': 2,
    'activation': 'softmax',
    'compile': {
        'loss': 'categorical_crossentropy',
        'optimizer': 'adam'
    },
    'train_config': {
        'batch_size': 256,
        'epochs': 10,
        'verbose': 1,
    },
}


def make_model(config):
    model = Sequential()
    model.add(Embedding(config['vocab_size'], 64))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(config['class_num'], activation=config['activation']))
    model.compile(loss=config['compile']['loss'], optimizer=config['compile']['optimizer'], metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于LSTM的文本分类')
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

    trainer = Trainer(model_name='classification_lstm', dataset=dataset, tokenizer=tokenizer, model=model)

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
