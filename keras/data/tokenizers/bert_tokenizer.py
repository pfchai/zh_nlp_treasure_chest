# -*- coding: utf-8 -*-

from typing import List

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding


class BertTokenizer():
    def __init__(self, dict_path):
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def encode(self, text1: str, text2: str, maxlen: int) -> list:
        return self.tokenizer.encode(text1, text2, maxlen=maxlen)

    def decode(self, ids, tokens=None):
        return self.tokenizer.decode(ids, tokens)

    def encode_X(self, texts: List, maxlen: int) -> List:
        batch_token_ids, batch_segment_ids = [], []
        for text in texts:
            if isinstance(text, str):
                text1 = text
                text2 = None
            if isinstance(text, list):
                text1, text2 = text[0], text[1]
            token_ids, segment_ids = self.tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return [batch_token_ids, batch_segment_ids]
