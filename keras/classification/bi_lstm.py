# -*- coding: utf-8 -*-

import warnings

import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences

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
        'epochs': 20,
        'verbose': 1,
    }
}


def make_model(config):
    model = Sequential()
    model.add(Embedding(config['vocab_size'], 64))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Dense(config['class_num'], activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    parser = create_parser(description='基于双向LSTM的文本分类')
    parser.add_argument('--checkpoint_name', help='测试的checkpoint 名称')
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

    trainer = Trainer(model_name='classification_bi_lstm', dataset=dataset, tokenizer=tokenizer, model=model)

    if args.do_train:
        trainer.train(config['train_config'])

    if args.do_test:
        model.load_weights(args.checkpoint_name)
        test_x, test_y = dataset.get_test_input_data(tokenizer)

        pred_result = model.predict(test_x)
        pred_y = pred_result.argmax(axis=1)
        test_y = test_y.argmax(axis=1)
        print(classification_report(test_y, pred_y))
