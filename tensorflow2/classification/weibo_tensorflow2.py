# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow_text import FastBertTokenizer


vocab = ['你', '好']
tf_tokenizer = FastBertTokenizer(vocab, token_out_type=tf.int64, lower_case_nfd_strip_accents=False)

tf_tokenizer.tokenize(['123'])
