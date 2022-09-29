# coding=utf-8

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from paddlenlp.datasets import load_dataset
from sklearn.metrics import classification_report

from zh_nlp_treasure_chest.utils import get_default_char_dict_path
from zh_nlp_treasure_chest.layers.tf_crf import CRF
from zh_nlp_treasure_chest.tokenization import CharTokenizer
from zh_nlp_treasure_chest.data.tf_ner_data_processor import DataProcessor


default_model_config = {
    'vocab_size': 7000,
    'embedding_dim': 128,
    'hidden_dim': 64,
    'label_num': 7,
}


def make_model(model_config=default_model_config):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(model_config['vocab_size'], model_config['embedding_dim'], mask_zero=True))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=model_config['hidden_dim'], return_sequences=True, recurrent_dropout=0.01)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(model_config['label_num'])))
    crf = CRF(model_config['label_num'], sparse_target=True)
    model.add(crf)
    return model, crf


def do_train(args, train_examples, valid_examples, data_processor):
    print('训练集样本数量： ', len(train_examples))
    print('训练集样本示例：')
    print(train_examples[1815])

    train_x = data_processor.encode_tokens([example['tokens'] for example in train_examples])
    train_y = data_processor.encode_labels([example['labels'] for example in train_examples])

    valid_x = data_processor.encode_tokens([example['tokens'] for example in valid_examples])
    valid_y = data_processor.encode_labels([example['labels'] for example in valid_examples])

    print('\n训练集模型输入数据示例')
    print(np.shape(train_x), np.shape(train_y))
    print(train_x[0][:10])
    print(train_y[0][:10])

    model, crf = make_model(default_model_config)
    optimizer = keras.optimizers.Adam(args.learning_rate)
    model.compile(optimizer=optimizer, loss=crf.loss, metrics=[crf.accuracy])
    model.summary()
    
    model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.num_train_epochs, validation_data=(valid_x, valid_y), verbose=1)
    
    return model


def do_test(model, test_examples, data_processor):
    input_tokens = [example['tokens'] for example in test_examples]
    # 模型有截断
    input_labels = [example['labels'][:data_processor.max_len] for example in test_examples]
    
    input_x = data_processor.encode_tokens(input_tokens)
    predict_results = model.predict(input_x)
    predict_ids = data_processor.predict_to_ids(predict_results, input_tokens)
    print(classification_report(np.hstack(input_labels), np.hstack(predict_ids), target_names=test_examples.label_list))


def do_predict(model, data_processor):
    while True:
        input_text = input('> ')
        if input_text.strip().lower() in ('q', 'quit', 'exit', 'bye'):
            break

        input_tokens = [list(input_text.strip())]
        input_x =  data_processor.encode_tokens(input_tokens)
        predict_results = model.predict(input_x)
        entities = data_processor.predict_to_entity(predict_results, input_tokens)[0]
        for entity in entities:
            print(entity)


def main(args):
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_name = os.path.join(args.output_dir, 'BiLSTM_CRF_{}.h5'.format(args.dataset))

    datasets = load_dataset(args.dataset)
    tokenizer = CharTokenizer(get_default_char_dict_path())
    data_processor = DataProcessor(tokenizer, datasets[0].label_list, args.max_seq_length)    

    model = None
    if args.do_train:
        # msra_ner 数据集包含训练集、验证集、测试集； peoples_daily_ner 无验证集，所以使用测试集当做验证集
        model = do_train(args, datasets[0], datasets[1], data_processor)
        model.save_weights(model_name)

    if args.do_test:
        if model is None:
            model, _ = make_model()
            model.load_weights(model_name)
        do_test(model, datasets[-1], data_processor)

    if args.do_predict:
        if model is None:
            model, _ = make_model()
            model.load_weights(model_name)
        do_predict(model, data_processor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='基于 LSTM + CRF 的命名实体识别')
    parser.add_argument(
        '--dataset', type=str, default='msra_ner',
        choices=('msra_ner', 'peoples_daily_ner'), help='指定数据集'
    )
    parser.add_argument("--output_dir", default='outputs', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="tokenization 之后序列最大长度。超过会被截断，小于会补齐")
    parser.add_argument("--batch_size", default=128, type=int, help="训练时一个 batch 包含多少条数据")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.", )
    # parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    # parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
    # parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)
    args = parser.parse_args()

    main(args)