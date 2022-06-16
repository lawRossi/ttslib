"""
@Author: Rossi
Created At: 2022-04-27
"""

from phkit.chinese import text2phoneme, pinyin2phoneme

from ttslib.text.symbols import extra, pinyin_valid_symbols


def text2phonemes(text):
    tokens = text2phoneme(text)
    phonemes = []
    group = []
    for token in tokens:
        if token not in extra:
            group.append(token)
        elif group:
            if len(group) != 3:
                print(text)
            assert len(group) == 3
            phonemes.append(group[0])
            phonemes.append(group[1] + group[2])
            group = []
    return phonemes


def pinyin2phonemes(text):
    pinyin_tokens = text.split(" ")
    pinyin_tokens = transform_erhua(pinyin_tokens)
    tokens = pinyin2phoneme(pinyin_tokens)
    phonemes = []
    group = []
    for token in tokens:
        if token not in extra:
            group.append(token)
        elif group:
            if len(group) != 3:
                print(group, text)
            assert len(group) == 3
            assert group[0] in pinyin_valid_symbols
            phonemes.append(group[0])
            merged_phoneme = group[1] + group[2]
            assert merged_phoneme in pinyin_valid_symbols
            phonemes.append(merged_phoneme)
            group = []
    return phonemes


def transform_erhua(pinyin_tokens):
    tokens = []
    for token in pinyin_tokens:
        if token[-2] == "r" and (token[-3] != "e" or len(token) > 3):
            tokens.append(token[:-2] + token[-1])
            tokens.append("er2")
        else:
            tokens.append(token)
    return tokens

