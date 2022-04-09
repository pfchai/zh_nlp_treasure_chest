# -*- coding: utf-8 -*-

from typing import List

from keras.preprocessing.sequence import pad_sequences


class CharTokenizer():
    def __init__(self, token_dict=None, token_dict_inv=None, vocabulary=None):
        if token_dict:
            self.token_dict = token_dict
            if token_dict_inv is None:
                self.token_dict_inv = {v: k for k, v in token_dict.items()}
        elif vocabulary:
            self.token_dict = vocabulary.token_dict
            self.token_dict_inv = vocabulary.token_dict_inv

        self.vocab_size = len(self.token_dict)

    def encode(self, text: str) -> list:
        tokens = []
        for ch in text:
            tokens.append(self.token_dict.get(ch, '0'))

        return tokens

    def decode(self, tokens: list) -> str:
        char_list = []
        for token in tokens:
            char_list.append(self.token_dict_inv.get(token, '[UNK]'))
        return ''.join(char_list)

    def encode_X(self, texts: List[str], maxlen: int) -> List:
        tokens = [self.encode(text) for text in texts]
        X = pad_sequences(tokens, maxlen=maxlen)
        return X
