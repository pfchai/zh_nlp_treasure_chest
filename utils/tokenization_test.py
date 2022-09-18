# coding=utf-8

import os
import tempfile
import tokenization
import unittest


class TokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing", ","
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            vocab_writer.write("".join(
                [x + "\n" for x in vocab_tokens]).encode("utf-8"))

        vocab_file = vocab_writer.name

        tokenizer = tokenization.FullTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize("UNwant\u00E9d,running")
        self.assertSequenceEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])

        self.assertSequenceEqual(
            tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        tokenizer = tokenization.BasicTokenizer()

        self.assertSequenceEqual(
            tokenizer.tokenize("ah\u535A\u63A8zz"),
            ["ah", "\u535A", "\u63A8", "zz"])

    def test_basic_tokenizer_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

        self.assertSequenceEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            ["hello", "!", "how", "are", "you", "?"])

        self.assertSequenceEqual(tokenizer.tokenize("H\u00E9llo"), ["hello"])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

        self.assertSequenceEqual(
            tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "),
            ["HeLLo", "!", "how", "Are", "yoU", "?"])

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

        self.assertSequenceEqual(tokenizer.tokenize(""), [])

        self.assertSequenceEqual(
            tokenizer.tokenize("unwanted running"),
            ["un", "##want", "##ed", "runn", "##ing"])

        self.assertSequenceEqual(
            tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    def test_convert_tokens_to_ids(self):
        vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
            "##ing"
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i

        self.assertSequenceEqual(
            tokenization.convert_tokens_to_ids(
                vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

    def test_is_whitespace(self):
        self.assertTrue(tokenization._is_whitespace(" "))
        self.assertTrue(tokenization._is_whitespace("\t"))
        self.assertTrue(tokenization._is_whitespace("\r"))
        self.assertTrue(tokenization._is_whitespace("\n"))
        self.assertTrue(tokenization._is_whitespace("\u00A0"))

        self.assertFalse(tokenization._is_whitespace("A"))
        self.assertFalse(tokenization._is_whitespace("-"))

    def test_is_control(self):
        self.assertTrue(tokenization._is_control("\u0005"))

        self.assertFalse(tokenization._is_control("A"))
        self.assertFalse(tokenization._is_control(" "))
        self.assertFalse(tokenization._is_control("\t"))
        self.assertFalse(tokenization._is_control("\r"))
        self.assertFalse(tokenization._is_control("\U0001F4A9"))

    def test_is_punctuation(self):
        self.assertTrue(tokenization._is_punctuation("-"))
        self.assertTrue(tokenization._is_punctuation("$"))
        self.assertTrue(tokenization._is_punctuation("`"))
        self.assertTrue(tokenization._is_punctuation("."))

        self.assertFalse(tokenization._is_punctuation("A"))
        self.assertFalse(tokenization._is_punctuation(" "))


if __name__ == "__main__":
    unittest.main()