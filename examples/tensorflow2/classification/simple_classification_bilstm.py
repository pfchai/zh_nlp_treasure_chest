# coding=utf-8

import os
import logging
import argparse
import datetime

# Disable Tensorflow log information 
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.captureWarnings(True)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from paddlenlp.datasets import load_dataset
from sklearn.metrics import classification_report

from zh_nlp_treasure_chest.utils import get_default_dict_path
from zh_nlp_treasure_chest.tokenization import FullTokenizer


global_config = {
    'vocab_size': 7000
}


parser = argparse.ArgumentParser(description='基于 双向 LSTM 的文本单标签分类模型')
parser.add_argument(
    '--dataset', type=str, default='chnsenticorp',
    choices=('chnsenticorp',), help='指定数据集'
)
parser.add_argument('--output_dir', default='outputs', type=str, help='模型训练中间结果和训练好的模型保存目录')
parser.add_argument('--max_seq_length', default=128, type=int, help='tokenization 之后序列最大长度。超过会被截断，小于会补齐')
parser.add_argument('--batch_size', default=128, type=int, help='训练时一个 batch 包含多少条数据')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Adam 优化器的学习率')
parser.add_argument('--add_special_tokens', default=True, type=bool, help='bert encode 时前后是否添加特殊token')
parser.add_argument('--num_train_epochs', default=2, type=int, help='训练时运行的 epoch 数量', )
parser.add_argument('--seed', type=int, default=42, help='随机数种子，保证结果可复现')
parser.add_argument('--do_train', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--do_predict', action='store_true', default=False)
args = parser.parse_args()


class DataProcessor():
    def __init__(self, tokenizer, input_max_len=128, num_classes=2) -> None:
        self.tokenizer = tokenizer
        self.max_len = input_max_len
        self.num_classes = num_classes

    def convert_text_to_input(self, texts):
        tokens = [self.tokenizer.tokenize(text) for text in texts]
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens]
        input_x = keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=self.max_len)
        return input_x

    def convert_label_to_input(self, labels):
        input_y = keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return input_y


def make_model(model_config=global_config):

    vocab_size = global_config['vocab_size']
    embedding_dim = 128

    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.5, recurrent_dropout=0.5)),
        keras.layers.Dense(embedding_dim, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])
    return model


def do_train(train_examples, valid_examples, data_processor):
    print('训练集样本数量： ', len(train_examples))
    print('训练集样本示例：')
    print(train_examples[0])

    train_x = data_processor.convert_text_to_input([example['text'] for example in train_examples])
    train_y = data_processor.convert_label_to_input([example['label'] for example in train_examples])

    print('\n训练集模型输入数据示例')
    print(np.shape(train_x), np.shape(train_y))
    print(train_x[0])
    print(train_y[0])

    if valid_examples is not None:
        valid_x = data_processor.convert_text_to_input([example['text'] for example in valid_examples])
        valid_y = data_processor.convert_label_to_input([example['label'] for example in valid_examples])

    model = make_model()
    optimizer = keras.optimizers.Adam(args.learning_rate)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    logdir = os.path.join(args.output_dir, 'log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    
    if valid_examples is not None:
        model.fit(
            train_x, train_y, batch_size=args.batch_size, epochs=args.num_train_epochs,
            validation_data=(valid_x, valid_y), verbose=1, callbacks=[tensorboard_callback]
        )
    else:
        model.fit(
            train_x, train_y, batch_size=args.batch_size, epochs=args.num_train_epochs,
            validation_split=0.2, verbose=1, callbacks=[tensorboard_callback]
        )
    return model


def do_test(model, test_examples, data_processor):

    test_x = data_processor.convert_text_to_input([example['text'] for example in test_examples])
    test_y = data_processor.convert_label_to_input([example['label'] for example in test_examples])

    pred_result = model.predict(test_x)
    pred_y = pred_result.argmax(axis=1)
    test_y = test_y.argmax(axis=1)
    print(classification_report(test_y, pred_y))


def do_predict(model, data_processor):
    while True:
        input_text = input('> ')
        if input_text.strip().lower() in ('q', 'quit', 'exit', 'bye'):
            break

        input_x =  data_processor.convert_text_to_input([input_text.strip()])
        predict_results = model.predict(input_x)
        print(predict_results[0])


def main():
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_name = os.path.join(args.output_dir, 'Simple_BiLSTM_{}.h5'.format(args.dataset))

    tokenizer = FullTokenizer(get_default_dict_path())
    datasets = load_dataset(args.dataset)
    data_processor = DataProcessor(tokenizer=tokenizer)
    
    global_config['vocab_size'] = len(tokenizer.vocab)
    print(global_config['vocab_size'])

    model = None
    if args.do_train:
        model = do_train(datasets[0], datasets[1], data_processor)
        model.save_weights(model_name)

    if args.do_test:
        if model is None:
            model = make_model()
            model.load_weights(model_name)

        # 
        do_test(model, datasets[1], data_processor)

    if args.do_predict:
        if model is None:
            model = make_model()
            model.load_weights(model_name)
        do_predict(model, data_processor)


if __name__ == '__main__':
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    main()