# -*- coding: utf-8 -*-

import re
import json
import warnings

import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences

from zh_nlp_demo.keras.data.vocabulary import Vocabulary
from zh_nlp_demo.keras.data.tokenizers.char_tokenizer import CharTokenizer


warnings.filterwarnings("ignore")
tf.compat.v1.set_random_seed(42)


def load_jsonl(file_name):
    examples = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            if 'label' in example:
                examples.append({
                    'text': example['text'],
                    'entities': [{
                        'label': label[2],
                        'start_offset': label[0],
                        'end_offset': label[1],
                    } for label in example['label']]
                })
            elif 'entities' in example:
                examples.append({
                    'text': example['text'],
                    'entities': example['entities']
                })

    return examples


def bio_to_entities(labels):
    entities = []
    for index, label in enumerate(labels):
        if label.startswith('B'):
            name = label[2:]
            entities.append({
                'label': name,
                'start_offset': index,
                'end_offset': index + 1,
            })
            continue
        if label.startswith('I'):
            if label[2:] != entities[-1]['label']:
                print('BIO 格式不规范')
                continue

            entities[-1]['end_offset'] = index + 1
            continue

    return entities


def entities_to_bio(entities, text_len):
    labels = ['O' for _ in range(text_len)]
    for entity in entities:
        is_overlap = False
        for index in range(entity['start_offset'], entity['end_offset']):
            if labels[index] != 'O':
                is_overlap = True
                break

        # 有重叠不处理、忽略后加入的重叠实体
        if is_overlap:
            continue

        labels[entity['start_offset']] = 'B-' + entity['label']
        for index in range(entity['start_offset'] + 1, entity['end_offset']):
            labels[index] = 'I-' + entity['label']
    return labels


def split_conversation_pro(examples):
    result = []
    for index, example in enumerate(examples):
        text, entities = example['text'], example['entities']
        labels = entities_to_bio(entities, len(text))
        assert len(text) == len(labels)

        for m in re.finditer(r'(【用户】： |【顾问】： ).*\n?', text):
            (s, e) = m.span()
            _text = m.group()
            role = 0
            if _text[1:3] == '用户':
                role = 1
            if _text[1:3] == '顾问':
                role = 2

            reply_type = 1
            m = re.match(r'.*<非文本消息 (\d+)>\n?', _text)
            if m:
                reply_type = int(m.groups()[0])

            # 有时候会多标记空格
            if s + 6 < len(labels):
                labels[s + 6] = re.sub(r'^I', 'B', labels[s + 6])
            result.append({
                'session_id': index,
                'role': role,
                'reply_type': reply_type,
                'text': text[s + 6: e - 1],
                'entities': bio_to_entities(labels[s + 6: e - 1])
            })

    return result


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


config = default_config

vocabulary = Vocabulary()
vocabulary.create_from_file()
tokenizer = CharTokenizer(vocabulary=vocabulary)
config['vocab_size'] = tokenizer.vocab_size

model = make_model(config)
# model.load_weights('outputs/checkpoints/best_model.hdf5')
model.load_weights('outputs/checkpoints/classification_bi_lstm_003_acc0.93.hdf5')


def predict(text):
    tokens = [tokenizer.encode(text)]
    X = pad_sequences(tokens, maxlen=200)

    pred_result = model.predict(X)
    pred_y = pred_result.argmax(axis=1)
    return pred_y[0], pred_result[0][1]


if __name__ == "__main__":
    examples = load_jsonl('/home/chaipf/work/ydl_nlp/ydl_nlp/models/event_extract/event_extract_800.jsonl')
    replies = split_conversation_pro(examples)
    # no_labels = list(filter(lambda reply: reply['session_id'] > 444, replies))
    no_labels = replies

    for reply in no_labels:
        if reply['role'] == 2:
            continue
        if reply['reply_type'] != 1:
            continue

        zh_text = ''.join(re.findall(r'[\u4e00-\u9fa5]+', reply['text']))
        if len(zh_text) < 3:
            reply['pred'] = 0
            reply['score'] = 0
            continue

        pred_y, score = predict(reply['text'])
        reply['pred'] = pred_y
        reply['score'] = score

    df = pd.DataFrame(no_labels)
    df.to_excel('预测.xlsx', index=False)
