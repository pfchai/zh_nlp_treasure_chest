# -*- coding: utf-8 -*-

from typing import List

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding


class BertTokenizer(Tokenizer):
    def __init__(self, dict_path):
        super(BertTokenizer, self).__init__(dict_path, do_lower_case=True)
        self.tokenizer_type = 'bert'

    def encode_X(self, texts: List, maxlen: int) -> List:
        batch_token_ids, batch_segment_ids = [], []
        for text in texts:
            if isinstance(text, str):
                text1 = text
                text2 = None
            if isinstance(text, list):
                text1, text2 = text[0], text[1]
            token_ids, segment_ids = self.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        return [batch_token_ids, batch_segment_ids]
