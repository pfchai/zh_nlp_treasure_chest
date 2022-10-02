# coding=utf-8

import os
import argparse


import numpy as np
import tensorflow as tf
from tensorflow import keras
from paddlenlp.datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    set_seed,
    TextClassificationPipeline,
)
from transformers import InputExample, InputFeatures

# 参考
bert_model_name = 'hfl/chinese-roberta-wwm-ext'


parser = argparse.ArgumentParser(description='基于 双向 Bert 的文本单标签分类模型')
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

def convert_example_to_input(examples, tokenizer, max_length=128):

    features = []
    for example in examples:
        inputs = tokenizer.encode_plus(
            example['text'], add_special_tokens=True,
            pad_to_max_length=True, truncation=True, max_length=max_length,
            return_token_type_ids=True, return_attention_mask=True,
        )
        input_ids, token_type_ids, attention_masks = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        labels = keras.utils.to_categorical([example['label']], num_classes=2)
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, label=labels[0]))

    def generate():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )
    return tf.data.Dataset.from_generator(
        generate,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        ),
    )


def make_model():
    config = AutoConfig.from_pretrained(bert_model_name)
    return TFAutoModelForSequenceClassification.from_pretrained(bert_model_name)


def do_train(train_examples, valid_examples, tokenizer):
    train_data = convert_example_to_input(train_examples, tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(1)

    for d in train_data:
        print(d)
        break

    validation_data = convert_example_to_input(valid_examples, tokenizer)
    validation_data = validation_data.batch(32)

    model = make_model()
    optimizer = keras.optimizers.Adam(args.learning_rate)
    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    model.fit(train_data, epochs=2, validation_data=validation_data)

    return model


def do_test(model, test_examples, data_processor):

    test_data = convert_example_to_input(train_examples, tokenizer)
    test_data = train_data.shuffle(100).batch(32)

    pred_result = model.predict(test_data)
    pred_y = pred_result.argmax(axis=1)
    test_y = test_y.argmax(axis=1)
    print(classification_report(test_y, pred_y))


def do_predict(pipe):
    while True:
        input_text = input('> ')
        if input_text.strip().lower() in ('q', 'quit', 'exit', 'bye'):
            break

        print(pipe(input_text.strip()))


def main():
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_name = os.path.join(args.output_dir, 'Simple_Bert_{}.h5'.format(args.dataset))

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(bert_model_name)
    datasets = load_dataset(args.dataset)

    model = None
    if args.do_train:
        model = do_train(datasets[0], datasets[1], tokenizer)
        tokenizer.save_pretrained(f"{args.output_dir}")
        model.save_pretrained(f"{args.output_dir}")

    if args.do_test:
        pass
        # if model is None:
        #     tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        #     model = TFAutoModelForSequenceClassification.from_pretrained(
        #         args.output_dir, id2label={0: 'negtive', 1: 'postive'}
        #     ) # modify labels as needed.
        # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        # do_test(pipe, datasets[1])

    if args.do_predict:
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model = TFAutoModelForSequenceClassification.from_pretrained(
                args.output_dir, id2label={0: 'negtive', 1: 'postive'}
            ) # modify labels as needed.
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        
        do_predict(pipe)


if __name__ == '__main__':
    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    set_seed(args.seed)

    main()