# -*- coding: utf-8 -*-


class CharTokenizer():
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.token_dict_inv = {v: k for k, v in token_dict.items()}

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
