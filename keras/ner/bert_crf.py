# -*- coding: utf-8 -*-

import os
import warnings

import keras
import tensorflow as tf
from keras.layers import Dense
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField

from zh_nlp_demo.keras.ner.utils import create_parser, create_dataset
from zh_nlp_demo.keras.trainer import Trainer
from zh_nlp_demo.keras.data.tokenizers.bert_tokenizer import BertTokenizer


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
tf.compat.v1.set_random_seed(42)


# 默认参数配置
default_config = {
    'bert_layers': 12,
    'categories': [],
    'crf_lr_multiplier': 1000,
    'compile': {
        'learning_rate': 2e-5,
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
    bert_layers = config['bert_layers']
    categories = config['categories']
    crf_lr_multiplier = config['crf_lr_multiplier']

    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
    output = bert.get_layer(output_layer).output
    output = Dense(len(categories) * 2 + 1)(output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = keras.models.Model(bert.input, output)

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(config['compile']['learning_rate']),
        metrics=[CRF.sparse_accuracy]
    )
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于Bert的命名实体识别')
    args = parser.parse_args()

    config = default_config

    # 数据集
    dataset = create_dataset(args)
    config = dataset.update_model_config(config)
    print('dataset load finished')

    tokenizer = BertTokenizer(dict_path)

    model = make_model(config)
    model.summary()

    trainer = Trainer(model_name='bert_crf', dataset=dataset, tokenizer=tokenizer, model=model)
    trainer.monitor = 'sparse_accuracy'

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        pass
