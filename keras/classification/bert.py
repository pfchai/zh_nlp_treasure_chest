# -*- coding: utf-8 -*-

import os
import warnings

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout
from bert4keras.models import build_transformer_model

from zh_nlp_demo.keras.classification.utils import create_parser
from zh_nlp_demo.keras.classification.utils import create_dataset
from zh_nlp_demo.keras.trainer import Trainer
from zh_nlp_demo.keras.data.tokenizers.bert_tokenizer import BertTokenizer


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
        'batch_size': 16,
        'epochs': 10,
        'verbose': 1,
    },
}


project_path = os.environ.get('ZH_NLP_DEMO_PATH')
pre_train_model_path = os.path.join(project_path, 'pre_train/tensorflow/Chinese_BERT_wwm/BERT_wwm_ext/')
config_path = os.path.join(pre_train_model_path, 'bert_config.json')
checkpoint_path = os.path.join(pre_train_model_path, 'bert_model.ckpt')
dict_path = os.path.join(pre_train_model_path, 'vocab.txt')


def make_model(config):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        return_keras_model=False,
    )

    output = Dropout(rate=0.1)(bert.model.output)
    output = Dense(
        units=config['class_num'], activation=config['activation'], kernel_initializer=bert.initializer
    )(output)
    model = keras.models.Model(bert.model.input, output)
    model.compile(loss=config['compile']['loss'], optimizer=config['compile']['optimizer'], metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于Bert的文本分类')
    args = parser.parse_args()

    config = default_config

    # 数据集
    dataset = create_dataset(args)
    config = dataset.update_model_config(config)
    print('dataset load finished')

    tokenizer = BertTokenizer(dict_path)

    model = make_model(config)
    model.summary()

    trainer = Trainer(model_name='lstm', dataset=dataset, tokenizer=tokenizer, model=model)

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
