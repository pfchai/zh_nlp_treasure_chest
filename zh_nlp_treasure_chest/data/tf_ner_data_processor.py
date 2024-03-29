# coding=utf-8

import numpy as np
from tensorflow import keras


class DataProcessor():

    def __init__(self, tokenizer, label_list, max_len=200) -> None:
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

        self.max_len = max_len if max_len else 200
        self.tokenizer = tokenizer

    def encode_tokens(self, texts_tokens):
        # texts_tokens = [list(text) for text in texts]
        input_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in texts_tokens]
        input_x = keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=self.max_len, padding='post')
        return input_x

    def encode_bert_tokens(self, texts_tokens, add_special_tokens=True):
        input_id_list, token_type_id_list, attention_mask_list = [], [], []

        for tokens in texts_tokens:
            inputs = self.tokenizer.encode_plus(tokens, add_special_tokens=add_special_tokens, truncation=True, max_length=self.max_len)
            input_ids, token_type_ids, attention_masks = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
            attention_mask_list.append(attention_masks)
            input_id_list.append(input_ids)
            token_type_id_list.append(token_type_ids)

        input_id_list = keras.preprocessing.sequence.pad_sequences(input_id_list, maxlen=self.max_len, padding='post')
        token_type_id_list = keras.preprocessing.sequence.pad_sequences(token_type_id_list, maxlen=self.max_len, padding='post')
        attention_mask_list = keras.preprocessing.sequence.pad_sequences(attention_mask_list, maxlen=self.max_len, padding='post')
        return [input_id_list, token_type_id_list, attention_mask_list]

    def encode_labels(self, labels):
        tag_seqs = keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.max_len, padding='post')
        input_y = np.asarray([keras.utils.to_categorical(seq, num_classes=len(self.label_list)) for seq in tag_seqs])
        return input_y

    def encode_bert_labels(self, labels, add_special_tokens=True, pad_token_label_id=6):
        if add_special_tokens:
            labels = [[pad_token_label_id] + ls[:self.max_len - 2] + [pad_token_label_id] for ls in labels]
        else:
            labels = [[pad_token_label_id] + ls[:self.max_len] + [pad_token_label_id] for ls in labels]
        tag_seqs = keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.max_len, padding='post')
        input_y = np.asarray([keras.utils.to_categorical(seq, num_classes=len(self.label_list)) for seq in tag_seqs])
        return input_y

    def tag_seq_to_entity(self, tokens, tag_seq):
        ret = []

        entity_tokens = []
        for token, tag in zip(tokens, tag_seq):
            if tag.startswith("B"):
                if entity_tokens:
                    ret.append([''.join(entity_tokens[0]), entity_tokens[1]])
                entity_tokens = [[token], tag.split('-')[1]]
            elif tag.startswith("I"):
                if not entity_tokens:
                    continue
                if  tag.split('-')[1] != entity_tokens[1]:
                    continue
                entity_tokens[0].append(token)
            elif tag == "O":
                if entity_tokens:
                    ret.append([''.join(entity_tokens[0]), entity_tokens[1]])
                    entity_tokens = []
        if entity_tokens:
            ret.append([''.join(entity_tokens[0]), entity_tokens[1]])
        return ret

    def predict_to_ids(self, predicts, input_tokens=None):
        id_seqs = []
        if input_tokens is not None:
            for predict, tokens in zip(predicts, input_tokens):
                id_seqs.append(np.argmax(predict, axis=-1)[:len(tokens)])
        else:
            id_seqs = np.argmax(predict, axis=-1)
        return id_seqs

    def predict_to_label(self, predicts, input_tokens=None):
        id_seqs = self.predict_to_ids(predicts, input_tokens)
        return [[self.id2label[i] for i in ids] for ids in id_seqs]

    def predict_to_entity(self, predicts, input_tokens):
        res = []
        tag_seqs = self.predict_to_label(predicts, input_tokens)
        for tokens, tag_seq in zip(input_tokens, tag_seqs):
            res.append(self.tag_seq_to_entity(tokens, tag_seq))
        return res