import pandas as pd
import jieba
import pickle
import re
from collections import Counter

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


def seg_words(contents, tokenize_style):
    string_segs = []
    if tokenize_style == "word":
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            segs = jieba.cut(content.strip())
            # print(" ".join(segs))
            string_segs.append(" ".join(segs))
    else:
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            # print(" ".join(list(content.strip())))
            string_segs.append(" ".join(list(content.strip())))
    return string_segs


def create_dict(string_train, label_list, path):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param string_train:经过分词的评论字符串列表，["" 吼吼 吼 ， 萌死 人 的 棒棒糖 ， 中 了 大众 点评 的 霸王餐"]
    :param label_list:用于保存各个评价对象的标签列表的字典
    :param path:存储生成的映射字典的路径
    :return:四个dict:word和label与索引index之间的双向映射字典
    """
    word_to_index = {}
    index_to_word = {}
    label_to_index = {0: 0, 1: 1, -1: -1, -2: -2}
    index_to_label = {0: 0, 1: 1, -1: -1, -2: -2}
    word_to_index[_PAD] = PAD_ID
    index_to_word[PAD_ID] = _PAD
    word_to_index[_UNK] = UNK_ID
    index_to_word[UNK_ID] = _UNK
    c_inputs = Counter()    # Counter用于统计字符串里某个字符出现的次数
    vocab_list = []  # 存储高词频的word及其相应的频数
    for string in string_train:
        c_inputs.update(string.split(" "))
        vocab_list = c_inputs.most_common(20000)  # 参数对word数量进行限制
    for i, word_freq in enumerate(vocab_list):
        # print(word_freq)  # word_freq是word和相应词频的元组
        word, _ = word_freq
        word_to_index[word] = i + 2
        index_to_word[i+2] = word
    with open(path, "wb") as dict_f:  # 创建映射字典后进行存储
        pickle.dump([word_to_index, index_to_word, label_to_index, index_to_label], dict_f)
    return word_to_index, index_to_word, label_to_index, index_to_label
