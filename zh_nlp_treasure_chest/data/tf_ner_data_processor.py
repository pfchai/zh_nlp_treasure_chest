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

    def encode_labels(self, labels):
        tag_seqs = keras.preprocessing.sequence.pad_sequences(labels, maxlen=self.max_len, padding='post')
        input_y = np.asarray([keras.utils.to_categorical(seq, num_classes=len(self.label_list)) for seq in tag_seqs])
        return input_y

    def _decode_predict(self, predict):
        tags = []
        for categorical in predict:
            tags.append(self.id2label[np.argmax(categorical)])
        return tags

    def decode_predict(self, predicts, input_tokens=None):
        res = []
        if input_tokens is not None:
            for predict, tokens in zip(predicts, input_tokens):
                res.append(self._decode_predict(predict[:len(tokens)]))
        else:
            for predict in predicts:
                res.append(self._decode_predict(predict))
        return res

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

    def predict_to_entity(self, input_tokens, predicts):
        res = []
        tag_seqs = self.decode_predict(predicts, input_tokens)
        for tokens, tag_seq in zip(input_tokens, tag_seqs):
            res.append(self.tag_seq_to_entity(tokens, tag_seq))
        return res