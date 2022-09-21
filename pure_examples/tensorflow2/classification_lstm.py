# -*- coding: utf-8 -*-
import os

from tensorflow import keras
from tensorflow_text import FastBertTokenizer
from paddlenlp.datasets import load_dataset
from sklearn.metrics import classification_report


[train_examples, dev_examples, test_examples] = load_dataset('chnsenticorp', splits=('train', 'dev', 'test'))

print('训练集样本数量： ', len(train_examples))
print('验证集样本数量： ', len(dev_examples))
print('测试集样本数量： ', len(test_examples))
print('训练集样本示例：')
print(train_examples[0])


def examples_to_ids(tokenizer, examples, maxlen=200, num_classes=2):
    input_ids = tokenizer.tokenize([example['text'] for example in examples]).to_list()
    input_x = keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=maxlen)
    input_y = keras.utils.to_categorical([example['label'] for example in examples], num_classes=num_classes)

    return input_x, input_y


vocab_dict_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data/dict/vocab.txt')
with open(vocab_dict_path, 'r', encoding='utf-8') as f:
    vocab = [line.strip() for line in f]

tokenizer = FastBertTokenizer(vocab)

train_x, train_y = examples_to_ids(tokenizer, train_examples)
dev_x, dev_y = examples_to_ids(tokenizer, train_examples)
test_x, test_y = examples_to_ids(tokenizer, train_examples)

print(train_x[0])
print(train_y[0])

vocab_size = len(vocab)
print('vocab_size', vocab_size)
embedding_dim = 128

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.5, recurrent_dropout=0.5)),
    keras.layers.Dense(embedding_dim, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

num_epochs = 20
history = model.fit(train_x, train_y, validation_data=(dev_x, dev_y), epochs=num_epochs, verbose=1)


pred_result = model.predict(test_x)
pred_y = pred_result.argmax(axis=1)
test_y = test_y.argmax(axis=1)
print(classification_report(test_y, pred_y))