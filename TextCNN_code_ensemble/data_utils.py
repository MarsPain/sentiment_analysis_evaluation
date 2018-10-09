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


stopwords_path = "data/stop_words.txt"


def seg_words(contents, tokenize_style):
    string_segs = []
    if tokenize_style == "word":
        stopwords = stopwordslist(stopwords_path)
        # stopwords_set = set(stopwords)
        stopwords_set = set()
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            segs = jieba.cut(content.strip())
            segs_new = []
            for word in segs:
                if word not in stopwords_set:
                    segs_new.append(word)
                else:
                    pass
                    # print("发现停用词：%s" % word)
            # print(" ".join(segs_new))
            string_segs.append(" ".join(segs_new))
    else:
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            # print(" ".join(list(content.strip())))
            string_segs.append(" ".join(list(content.strip())))
    return string_segs


def stopwordslist(path):
    # stopwords = [line.strip() for line in open(path, 'r', encoding='utf-8').readlines()]
    stopwords = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    # print(stopwords)
    return stopwords


def create_dict(string_train, label_list, path, vocab_size):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param string_train:经过分词的评论字符串列表，["" 吼吼 吼 ， 萌死 人 的 棒棒糖 ， 中 了 大众 点评 的 霸王餐"]
    :param label_list:用于保存各个评价对象的标签列表的字典
    :param path:存储生成的映射字典的路径
    :param vocab_size: word到index的映射字典的大小
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
        vocab_list = c_inputs.most_common(vocab_size)  # 参数对word数量进行限制
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


# def get_label_pert(train_data_df, columns):
#     len_data = train_data_df.shape[0]
#     label_pert_dict = {}
#     for column in columns[2:]:
#         label_list = list(train_data_df[column])
#         label_1_true = 0
#         label_0 = 0
#         label_1_false = 0
#         label_2 = 0
#         for label in label_list:
#             if label == 1:
#                 label_1_true += 1
#             elif label == -1:
#                 label_1_false += 1
#             elif label == -2:
#                 label_2 += 1
#             else:
#                 label_0 += 1
#         label_1_true_pert = label_1_true/len_data
#         label_1_false_pert = label_1_false/len_data
#         label_2_pert = label_2/len_data
#         label_0_pert = label_0/len_data
#         # print("label_pert(1:0:-1:-2):", column, label_1_true_pert, label_0_pert, label_1_false_pert,
#         #       label_2_pert)
#         label_pert_dict[column] = [label_1_true_pert, label_0_pert, label_1_false_pert, label_2_pert]
#     return label_pert_dict


# def get_labal_weight(label_pert_dict):
#     label_weight_dict = {}
#     for column, label_pert in label_pert_dict.items():
#         label_weight = [1-label_pert[0], 1-label_pert[1], 1-label_pert[2], 1-label_pert[3]]
#         label_weight_dict[column] = label_weight
#     return label_weight_dict


def get_labal_weight(label_dict, columns, num_classes):
    len_data = len(label_dict[columns[2]])
    print("len_data:", len_data)
    label_weight_dict = {}
    for column in columns[2:]:
        label_list = list(label_dict[column])
        # print("label_list:", label_list)
        label_0 = 0
        label_1 = 0
        label_2 = 0
        label_3 = 0
        for label in label_list:
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            else:
                label_3 += 1
        # label_number_array = np.asarray([label_0, label_1, label_2, label_3])
        # label_weight_list = len_data / (num_classes * label_number_array)
        label_weight_list = [1, 1, 1, 1]    # 取消权重参数
        # print(column, label_number_array, label_weight_list)
        label_weight_dict[column] = label_weight_list
    return label_weight_dict


def get_least_label(label_dict, columns):
    least_label_dict = {}
    for column in columns[2:]:
        label_list = list(label_dict[column])
        label_0 = 0
        label_1 = 0
        label_2 = 0
        label_3 = 0
        for label in label_list:
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            else:
                label_3 += 1
        label_number_array = np.asarray([label_0, label_1, label_2, label_3])
        least_label_num = min(label_0, label_1, label_2, label_3)
        print(column, label_number_array, least_label_num)
        least_label_dict[column] = least_label_num
    return least_label_dict


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


class SampleBatchManager:
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
            label = data[2][i*batch_size:(i+1)*batch_size]
            batch_data.append([sentences, vector_tfidf, label])
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
        if label == 0:
            weights_list_batch[i] = weights_dict[0]
        elif label == 1:
            weights_list_batch[i] = weights_dict[1]
        elif label == 2:
            weights_list_batch[i] = weights_dict[2]
        else:
            weights_list_batch[i] = weights_dict[3]
    return weights_list_batch


def compute_confuse_matrix(logit, label, small_value):
    length = len(label)
    true_positive_0 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_0 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_0 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_1 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_1 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_1 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_2 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_2 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_2 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_3 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_3 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_3 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    for i in range(length):
        predict = np.argmax(logit[i])
        # 用于计算0的精确率和召回率
        if label[i] == 0 and predict == 0:
            true_positive_0 += 1
        elif (label[i] == 1 or label[i] == 2 or label[i] == 3) and predict == 0:
            false_positive_0 += 1
        elif label[i] == 0 and (predict == 1 or predict == 2 or predict == 3):
            false_negative_0 += 1
        # 用于计算1的精确率和召回率
        if label[i] == 1 and predict == 1:
            true_positive_1 += 1
        elif (label[i] == 0 or label[i] == 2 or label[i] == 3) and predict == 1:
            false_positive_1 += 1
        elif label[i] == 1 and (predict == 0 or predict == 2 or predict == 3):
            false_negative_1 += 1
        # 用于计算2的精确率和召回率
        if label[i] == 2 and predict == 2:
            true_positive_2 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 3) and predict == 2:
            false_positive_2 += 1
        elif label[i] == 2 and (predict == 0 or predict == 1 or predict == 3):
            false_negative_2 += 1
        # 用于计算3的精确率和召回率
        if label[i] == 3 and predict == 3:
            true_positive_3 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 2) and predict == 3:
            false_positive_3 += 1
        elif label[i] == 3 and (predict == 0 or predict == 1 or predict == 2):
            false_negative_3 += 1
    return true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1,\
           true_positive_2, false_positive_2, false_negative_2, true_positive_3, false_positive_3, false_negative_3


def afresh_sampling(train_data, least_label_dict, column_name, batch_size):
    sentences, feature_vector, label_dict = train_data
    label = label_dict[column_name]
    sentences_sample = []
    feature_vector_sample = []
    label_sample = []
    least_label_num = least_label_dict[column_name]
    len_data = len(sentences)
    label_0_count = 0
    label_1_count = 0
    label_2_count = 0
    label_3_count = 0
    random_perm = np.random.permutation(len_data)   # 对索引进行随机排序
    for index in random_perm:
        if label_0_count == least_label_num and label_1_count == least_label_num and label_2_count == least_label_num and label_3_count == least_label_num:
            break
        if label[index] == 0 and label_0_count != least_label_num:
            sentences_sample.append(sentences[index])
            feature_vector_sample.append(feature_vector[index])
            label_sample.append(label[index])
            label_0_count += 1
        elif label[index] == 1 and label_1_count != least_label_num:
            sentences_sample.append(sentences[index])
            feature_vector_sample.append(feature_vector[index])
            label_sample.append(label[index])
            label_1_count += 1
        elif label[index] == 2 and label_2_count != least_label_num:
            sentences_sample.append(sentences[index])
            feature_vector_sample.append(feature_vector[index])
            label_sample.append(label[index])
            label_2_count += 1
        elif label[index] == 3 and label_3_count != least_label_num:
            sentences_sample.append(sentences[index])
            feature_vector_sample.append(feature_vector[index])
            label_sample.append(label[index])
            label_3_count += 1
        else:
            continue
    # print("各标签数量：", label_0_count, label_1_count, label_2_count, label_3_count)
    # 对重采样的数据再次进行随机排序
    sentences_sample_shuffle = []
    vector_tfidf_sample_shuffle = []
    label_sample_shuffle = []
    random_perm_sample = np.random.permutation(4 * least_label_num)   # 对索引进行随机排序
    for index in random_perm_sample:
        sentences_sample_shuffle.append(sentences_sample[index])
        vector_tfidf_sample_shuffle.append(feature_vector_sample[index])
        label_sample_shuffle.append(label_sample[index])
    train_data_sample_shuffle = [sentences_sample_shuffle, vector_tfidf_sample_shuffle, label_sample_shuffle]
    train_batch_sample_manager = SampleBatchManager(train_data_sample_shuffle, batch_size)
    return train_batch_sample_manager
