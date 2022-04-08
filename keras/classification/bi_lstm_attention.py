# -*- coding: utf-8 -*-

import warnings

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout

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
    'train_config': {
        'batch_size': 256,
        'epochs': 10,
        'validation_split': 0.1,
        'verbose': 1,
    }
}


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(
            name='att_weight', shape=(hidden_size, self.attention_size),
            initializer='uniform', trainable=True
        )
        self.b = self.add_weight(
            name='att_bias', shape=(self.attention_size,),
            initializer='uniform', trainable=True
        )
        self.V = self.add_weight(
            name='att_var', shape=(self.attention_size,),
            initializer='uniform', trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def make_model(config):
    model = Sequential()
    model.add(Embedding(config['vocab_size'], 64))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
    model.add(AttentionLayer(64))
    model.add(Dropout(0.2))
    model.add(Dense(config['class_num'], activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于双向LSTM + Attention的文本分类')
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

    trainer = Trainer(model_name='bi-lstm', dataset=dataset, tokenizer=tokenizer, model=model)

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
