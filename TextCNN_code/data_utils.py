import pandas as pd
import jieba
import pickle
import numpy as np
import math
import re
from collections import Counter
import random

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
    label_to_index = {1: 0, 0: 1, -1: 2, -2: 3}
    index_to_label = {0: 1, 1: 0, 2: -1, 3: -2}
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


def get_vector_tfidf(string, tfidf_dict):
    vector_tfidf_list = []
    for s in string:
        vector_tfidf = []
        word_list = s.split(" ")
        for word in word_list:
            if word in tfidf_dict:
                vector_tfidf.append(tfidf_dict[word])
            else:
                vector_tfidf.append(1)
        vector_tfidf_list.append(vector_tfidf)
    return vector_tfidf_list


def get_label_pert(train_data_df, columns):
    len_data = train_data_df.shape[0]
    label_pert_dict = {}
    for column in columns[2:]:
        label_list = list(train_data_df[column])
        label_1_true = 0
        label_0 = 0
        label_1_false = 0
        label_2 = 0
        for label in label_list:
            if label == 1:
                label_1_true += 1
            elif label == -1:
                label_1_false += 1
            elif label == -2:
                label_2 += 1
            else:
                label_0 += 1
        label_1_true_pert = label_1_true/len_data
        label_1_false_pert = label_1_false/len_data
        label_2_pert = label_2/len_data
        label_0_pert = label_0/len_data
        # print("label_pert(1:0:-1:-2):", column, label_1_true_pert, label_0_pert, label_1_false_pert,
        #       label_2_pert)
        label_pert_dict[column] = [label_1_true_pert, label_0_pert, label_1_false_pert, label_2_pert]
    return label_pert_dict


def get_labal_weight(label_pert_dict):
    label_weight_dict = {}
    for column, label_pert in label_pert_dict.items():
        label_weight = [1-label_pert[0], 1-label_pert[1], 1-label_pert[2], 1-label_pert[3]]
        # sum_pert = sum(label_weight)
        # label_weight = [0.5+label_weight[0]/sum_pert, 0.5+label_weight[1]/sum_pert,
        #                 0.5+label_weight[2]/sum_pert, 0.5+label_weight[3]/sum_pert]
        # print("label_weight(1:0:-1:-2):", label_weight)
        label_weight_dict[column] = label_weight
    return label_weight_dict


def sentence_word_to_index(string, word_to_index, label_train_dict, label_to_index):
    sentences = []
    for s in string:
        # print(s)
        word_list = s.split(" ")
        # word_to_index只保存了预先设置的词库大小，所以没存储的词被初始化为UNK_ID
        sentence = [word_to_index.get(word, UNK_ID) for word in word_list]
        # print(sentence)
        if len(word_list) != len(sentence):
            print("Error!!!!!!!!!", len(word_list), len(sentence))
        sentences.append(sentence)
    # print("sentences:", sentences)
    label_train_dict_new = {}
    for column, label_list in label_train_dict.items():
        label_train_dict_new[column] = []
    for column, label_list in label_train_dict.items():
        for label in label_list:
            label_train_dict_new[column].append(label_to_index[label])
    return sentences, label_train_dict_new


def shuffle_padding(sentences, feature_vector, label_dict, max_len):
    sentences_shuffle = []
    label_dict_shuffle = {}
    for column, label_list in label_dict.items():
        label_dict_shuffle[column] = []
    vector_tfidf_shuffle = []
    len_data = len(sentences)
    random_perm = np.random.permutation(len_data)   # 对索引进行随机排序
    for index in random_perm:
        if len(sentences[index]) != len(feature_vector[index]):
            print("Error!!!!!!", len(sentences[index]), len(feature_vector))
        sentences_shuffle.append(sentences[index])
        vector_tfidf_shuffle.append(feature_vector[index])
        for column, label_list in label_dict.items():
            label_dict_shuffle[column].append(label_list[index])
    sentences_padding = pad_sequences(sentences_shuffle, max_len, PAD_ID)
    # print(sentences_padding[0])
    vector_tfidf_padding = pad_sequences(vector_tfidf_shuffle, max_len, PAD_ID)
    # print(vector_tfidf_padding[0])
    data = [sentences_padding, vector_tfidf_padding, label_dict_shuffle]
    return data


def get_max_len(sentences):
    max_len = 0
    for sentence in sentences:
        max_len = max(max_len, len(sentence))
    return max_len


def pad_sequences(sequence, max_len, PAD_ID):
    sequence_padding = []
    for string in sequence:
        if len(string) < max_len:
            padding = [PAD_ID] * (max_len - len(string))
            sequence_padding.append(string + padding)
        elif len(string) > max_len:
            sequence_padding.append(string[:max_len])
        else:
            sequence_padding.append(string)
    return sequence_padding


class BatchManager:
    """
    用于生成batch数据的batch管理类
    """
    def __init__(self, data,  batch_size):
        self.batch_data = self.get_batch(data, batch_size)
        self.len_data = len(self.batch_data)

    @staticmethod
    def get_batch(data, batch_size):
        num_batch = int(math.ceil(len(data[0]) / batch_size))
        batch_data = []
        for i in range(num_batch):
            sentences = data[0][i*batch_size:(i+1)*batch_size]
            vector_tfidf = data[1][i*batch_size:(i+1)*batch_size]
            label_dict = data[2]
            label_dict_mini_batch = {}
            for column, label_list in label_dict.items():
                label_dict_mini_batch[column] = label_list[i*batch_size:(i+1)*batch_size]
            batch_data.append([sentences, vector_tfidf, label_dict_mini_batch])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def get_weights_for_current_batch(answer_list, weights_dict):
    weights_list_batch = list(np.ones((len(answer_list))))
    answer_list = list(answer_list)
    for i, label in enumerate(answer_list):
        if label == 1:
            weights_list_batch[i] = weights_dict[0]
        elif label == 0:
            weights_list_batch[i] = weights_dict[1]
        elif label == -1:
            weights_list_batch[i] = weights_dict[2]
        else:
            weights_list_batch[i] = weights_dict[3]
    return weights_list_batch


def compute_confuse_matrix(logit, label):
    length = len(label)
    true_positive = 0  # TP:if label is true('0/2/3'), and predict is true('0/2/3')
    false_positive = 0  # FP:if label is false('1'),but predict is ture('0/2/3')
    true_negative = 0  # TN:if label is false('0'),and predict is false('0')
    false_negative = 0  # FN:if label is false('0/2/3'),but predict is true('1')
    for i in range(length):
        predict = np.argmax(logit[i])
        # print(predict, label[i])
        if (predict == 0 and label[i] == 0) or (predict == 2 and label[i] == 2) or \
                (predict == 3 and label[i] == 3):
            true_positive += 1
        elif predict == 1 and (label[i] == 0 or label[i] == 2 or label[i] == 3):
            # print("yes")
            false_positive += 1
        elif predict == 0 and label[i] == 0:
            # print("yes")
            true_negative += 1
        elif (predict == 0 or label[i] == 2 or label[i] == 3) and label[i] == 1:
            # print("yes")
            false_negative += 1
        # elif predict == 3 and label[i] == 0:
        #     pass
            # print("yes!!!!!!")
    return true_positive, false_positive, true_negative, false_negative
