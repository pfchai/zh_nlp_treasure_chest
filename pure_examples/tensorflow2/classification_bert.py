# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from paddlenlp.datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer


bert_name = 'hfl/chinese-roberta-wwm-ext'
# bert_name = 'bert-base-chinese'

num_labels=2
max_length=200
batch_size = 16
num_epochs = 2

[train_examples, dev_examples, test_examples] = load_dataset('chnsenticorp', splits=('train', 'dev', 'test'))

print('训练集样本数量： ', len(train_examples))
print('验证集样本数量： ', len(dev_examples))
print('测试集样本数量： ', len(test_examples))
print('训练集样本示例：')
print(train_examples[0])

tokenizer = AutoTokenizer.from_pretrained(bert_name)


# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def data_preprocess(examples):
    bert_input = tokenizer([e['text'] for e in examples], max_length=max_length, pad_to_max_length=True, return_attention_mask=True)
    labels = [e['label'] for e in examples]
    return tf.data.Dataset.from_tensor_slices((bert_input['input_ids'], bert_input['token_type_ids'], bert_input['attention_mask'], labels)).map(map_example_to_dict)


train_dataset = data_preprocess(train_examples).batch(batch_size)
dev_dataset = data_preprocess(dev_examples).batch(batch_size)
test_dataset = data_preprocess(test_examples).batch(batch_size)

model = TFAutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels)

batches_per_epoch = len(train_examples) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)

optimizer, _ = create_optimizer(
    init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

history = model.fit(
    train_dataset,
    validation_data=dev_dataset,
    epochs=num_epochs,
)

print("# evaluate test_set:", model.evaluate(dev_dataset))


# 模型推理
sentences = [example['text'] for example in test_examples[:8]]
tokenized = tokenizer(sentences, return_tensors="np", padding="longest")

outputs = model(tokenized).logits
classifications = np.argmax(outputs, axis=1)
print(classifications)
