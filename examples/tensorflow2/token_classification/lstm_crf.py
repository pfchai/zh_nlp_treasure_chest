# coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from paddlenlp.datasets import load_dataset

from zh_nlp_treasure_chest.utils import get_default_char_dict_path
from zh_nlp_treasure_chest.layers.tf_crf import CRF
from zh_nlp_treasure_chest.tokenization import CharTokenizer
from zh_nlp_treasure_chest.data.tf_ner_data_processor import DataProcessor


keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


[train_examples, test_examples] = load_dataset('msra_ner', splits=('train', 'test'))

print('训练集样本数量： ', len(train_examples))
print('测试集样本数量： ', len(test_examples))

print('训练集样本示例：')
print(train_examples[1815])


tokenizer = CharTokenizer(get_default_char_dict_path())
data_processor = DataProcessor(tokenizer, train_examples.label_list, 200)

train_x = data_processor.encode_tokens([example['tokens'] for example in train_examples])
train_y = data_processor.encode_labels([example['labels'] for example in train_examples])

test_x = data_processor.encode_tokens([example['tokens'] for example in test_examples])
test_y = data_processor.encode_labels([example['labels'] for example in test_examples])


print('\n训练集模型输入数据示例')
print(np.shape(train_x), np.shape(train_y))
print(train_x[0][:10])
print(train_y[0][:10])

print('\n测试集模型输入数据示例')
print(np.shape(test_x), np.shape(test_y))
print(test_x[0][:10])
print(test_y[0][:10])

vocab_size = len(tokenizer.vocab)
embedding_dim = 128
hidden_dim = 64
label_num = len(train_examples.label_list)
batch_size = 128
epochs = 3

print('vocab size ', vocab_size)


model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_dim, return_sequences=True, recurrent_dropout=0.01)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(label_num)))
crf = CRF(label_num, sparse_target=True)
model.add(crf)
model.compile(optimizer='adam', loss=crf.loss, metrics=[crf.accuracy])
model.summary()


model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_x, test_y), verbose=1)
model.save_weights('lstm_crf.h5')

input_tokens = [example['tokens'] for example in test_examples[:10]]
input_x = data_processor.encode_tokens(input_tokens)
predict_results = model.predict(input_x)
predict_entities = data_processor.predict_to_entity(input_tokens, predict_results)

for tokens, entities in zip(input_tokens, predict_entities):
    print(''.join(tokens))
    for entity in entities:
        print(entity)
    print('--' * 50)