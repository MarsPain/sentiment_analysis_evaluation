from Attention_RCNN import config
import logging
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import math
import random
from Attention_RCNN.data_utils import seg_words, get_vector_tfidf, get_vector_tfidf_from_dict
from Attention_RCNN.utils import load_data_from_csv, load_tfidf_dict,\
    load_word_embedding, get_tfidf_and_save
from Attention_RCNN.model import TextCNN
from Attention_RCNN.confidence_adjust import adjust_confidence, automatic_search

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"

# test_data_path = "../data/sentiment_analysis_testa.csv"
test_data_path = "../data/sentiment_analysis_validationset.csv"
# test_data_path = "result.csv"
test_data_pkl = "pkl/test_data.pkl"
test_data_predict_out_path = "result.csv"
models_dir = "ckpt_4"
word_label_dict = "pkl/word_label_dict.pkl"
tfidf_path = "data/tfidf.txt"
tfidf_pkl_path = "data/tfidf.pkl"
idf_path = "data/idf_4_traffic.txt"
log_predict_error_dir = "error_log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def get_data():
    # load data
    logger.info("start load data")
    test_data_df = load_data_from_csv(test_data_path)
    if os.path.exists(test_data_pkl):    # 若train_valid_test已被处理和存储
        with open(word_label_dict, 'rb') as dict_f:
            word_to_index, index_to_word, label_to_index, index_to_label = pickle.load(dict_f)
        with open(test_data_pkl, 'rb') as f:
            test_data = pickle.load(f)
            test_data = test_data[0]
    else:
        # seg words
        logger.info("start seg test data")
        content_test = test_data_df.iloc[:, 1]
        string_test = seg_words(content_test, config.tokenize_style)
        logger.info("complete seg test data")
        with open(word_label_dict, 'rb') as dict_f:
            word_to_index, index_to_word, label_to_index, index_to_label = pickle.load(dict_f)
        tfidf_dict = load_tfidf_dict(tfidf_path)
        test_vector_tfidf = get_vector_tfidf_from_dict(string_test, tfidf_dict)
        # idf_dict = load_tfidf_dict(idf_path)
        # test_vector_tfidf = get_vector_tfidf_from_dict(string_test, idf_dict)
        # # 获取tfidf模型以及已被排序的字典
        # if not os.path.exists(tfidf_pkl_path):
        #     vectorizer_tfidf, word_list_sort_dict = get_tfidf_and_save(string_test, tfidf_pkl_path, config.tokenize_style)
        # else:
        #     with open(tfidf_pkl_path, "rb") as f:
        #         vectorizer_tfidf, word_list_sort_dict = pickle.load(f)
        # # 根据tfidf模型以及已被排序的字典获取训练集和验证集的tfidf值向量作为额外的特征向量
        # test_vector_tfidf = get_vector_tfidf(string_test, vectorizer_tfidf, word_list_sort_dict)
        sentences_test = sentence_word_to_index(string_test, word_to_index)
        sentences_padding = padding_data(sentences_test, config.max_len)
        vector_tfidf_padding = padding_data(test_vector_tfidf, config.max_len)
        # idf_attention_padding = []
        # for i in range(len(vector_tfidf_padding)):
        #     vector_tfidf = vector_tfidf_padding[i]
        #     vector_tfidf = np.asarray(vector_tfidf)
        #     # print("vector_tfidf", vector_tfidf)
        #     idf_attention = np.reshape(vector_tfidf, [-1, 1])
        #     idf_attention = idf_attention.tolist()
        #     # print("idf_attention", idf_attention)
        #     idf_attention_padding.append(idf_attention)
        test_data = [sentences_padding, vector_tfidf_padding]
        with open(test_data_pkl, "wb") as f:
            pickle.dump([test_data], f)
    print(len(test_data[0]), len(test_data[1]))
    test_batch_manager = BatchManager(test_data, int(config.batch_size))
    logger.info("complete load data")
    return test_data_df, test_batch_manager, index_to_word, index_to_label, label_to_index


def sentence_word_to_index(string, word_to_index):
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
    return sentences


def padding_data(sequence, max_len):
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
            batch_data.append([sentences, vector_tfidf])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def create_model(sess, index_to_word):
    text_cnn = TextCNN()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    return text_cnn, saver


def predict():
    # model_name = get_parer()
    test_data_df, test_batch_manager, index_to_word, index_to_label, label_to_index = get_data()
    columns = test_data_df.columns.tolist()
    # model predict
    logger.info("start predict test data")
    column = columns[config.column_index]  # 选择评价对象
    model_path = os.path.join(models_dir, column)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        logits_all = []
        text_cnn, saver = create_model(sess, index_to_word)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        logger.info("compete load %s model and start predict" % column)
        for batch in test_batch_manager.iter_batch(shuffle=False):
            test_x, features_vector = batch
            # print(len(test_x[0]), test_x[0])
            feed_dict = {text_cnn.input_x: test_x, text_cnn.features_vector: features_vector,
                         text_cnn.dropout_keep_prob: 1.0}
            logits = sess.run([text_cnn.logits], feed_dict)
            # print("logits:", logits[0])
            for i in range(len(logits[0])):
                logits_all.append(list(logits[0][i]))
            # print(logits_all)
        predictions_all = logits_to_predictions(logits_all, column, label_to_index)  # 将logits转化为predictions
        # 对比predictions和真实label，如果不对，就打印logits到文件中
        # write_predict_error_to_file(predictions_all, logits_all, column, label_to_index, log_predict_error_dir)
        # print(len(predictions_all))
        _ = test_f_score_in_valid_data(predictions_all, column, label_to_index)  # test_f_score_in_valid_data
        # 将predictions映射到label，预测得到的是label的index。
        logger.info("start transfer index to label")
        for i in range(len(predictions_all)):
            predictions_all[i] = index_to_label[predictions_all[i]]
        # print(predictions_all)
        test_data_df[column] = predictions_all
    logger.info("compete %s predict" % column)
    test_data_df.to_csv(test_data_predict_out_path, encoding="utf_8_sig", index=False)


def write_predict_error_to_file(predictions_all, logits_all, column_name, label_to_index, error_dir):
    validate_data_df = load_data_from_csv(test_data_path)
    label_valid = list(validate_data_df[column_name].iloc[:])
    for i in range(len(predictions_all)):
        label_valid[i] = label_to_index[label_valid[i]]  # 获取真实的标签
    label_valid_list = []
    predictions_all_list = []
    logit_0_all_list = []
    logit_1_all_list = []
    logit_2_all_list = []
    logit_3_all_list = []
    error_path = os.path.join(error_dir, column_name + ".csv")
    for i in range(len(predictions_all)):
        if label_valid[i] != predictions_all[i]:
            label_valid_list.append(label_valid[i])
            predictions_all_list.append(predictions_all[i])
            logit_0_all_list.append(logits_all[i][0])
            logit_1_all_list.append(logits_all[i][1])
            logit_2_all_list.append(logits_all[i][2])
            logit_3_all_list.append(logits_all[i][3])
    label_valid_array = pd.Series(label_valid_list, name="正确标签")
    predictions_all_array = pd.Series(predictions_all_list, name="错误标签")
    logit_0_all_array = pd.Series(logit_0_all_list, name="标签0的置信度")
    logit_1_all_array = pd.Series(logit_1_all_list, name="标签1的置信度")
    logit_2_all_array = pd.Series(logit_2_all_list, name="标签2的置信度")
    logit_3_all_array = pd.Series(logit_3_all_list, name="标签3的置信度")
    error_log_array_list = [label_valid_array, predictions_all_array, logit_0_all_array, logit_1_all_array, logit_2_all_array, logit_3_all_array]
    error_log_df = pd.concat(error_log_array_list, axis=1)
    error_log_df.to_csv(error_path, encoding="utf-8")


def test_f_score_in_valid_data(predictions_all, column_name, label_to_index):
    validate_data_df = load_data_from_csv(test_data_path)
    label_valid = list(validate_data_df[column_name].iloc[:])
    for i in range(len(predictions_all)):
        label_valid[i] = label_to_index[label_valid[i]]
    # print("predictions_all:", len(predictions_all), predictions_all)
    # print("label_valid:", len(label_valid), label_valid)
    f_0, f_1, f_2, f_3 = compute_confuse_matrix(predictions_all, label_valid, 0.00001)
    print("f_0, f_1, f_2, f_3:", f_0, f_1, f_2, f_3)
    print("f1_score:", (f_0 + f_1 + f_2 + f_3) / 4)
    return (f_0 + f_1 + f_2 + f_3) / 4


def compute_confuse_matrix(predictions_all, label, small_value):
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
        # 用于计算0的精确率和召回率
        if label[i] == 0 and predictions_all[i] == 0:
            true_positive_0 += 1
        elif (label[i] == 1 or label[i] == 2 or label[i] == 3) and predictions_all[i] == 0:
            false_positive_0 += 1
        elif label[i] == 0 and (predictions_all[i] == 1 or predictions_all[i] == 2 or predictions_all == 3):
            false_negative_0 += 1
        # 用于计算1的精确率和召回率
        if label[i] == 1 and predictions_all[i] == 1:
            true_positive_1 += 1
        elif (label[i] == 0 or label[i] == 2 or label[i] == 3) and predictions_all[i] == 1:
            false_positive_1 += 1
        elif label[i] == 1 and (predictions_all[i] == 0 or predictions_all[i] == 2 or predictions_all[i] == 3):
            false_negative_1 += 1
        # 用于计算2的精确率和召回率
        if label[i] == 2 and predictions_all[i] == 2:
            true_positive_2 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 3) and predictions_all[i] == 2:
            false_positive_2 += 1
        elif label[i] == 2 and (predictions_all[i] == 0 or predictions_all[i] == 1 or predictions_all[i] == 3):
            false_negative_2 += 1
        # 用于计算3的精确率和召回率
        if label[i] == 3 and predictions_all[i] == 3:
            true_positive_3 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 2) and predictions_all[i] == 3:
            false_positive_3 += 1
        elif label[i] == 3 and (predictions_all[i] == 0 or predictions_all[i] == 1 or predictions_all[i] == 2):
            false_negative_3 += 1
    p_0 = float(true_positive_0)/float(true_positive_0+false_positive_0+small_value)
    r_0 = float(true_positive_0)/float(true_positive_0+false_negative_0+small_value)
    # print("标签0的预测情况：", true_positive_0, false_positive_0, false_negative_0, p_0, r_0)
    f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + small_value)
    p_1 = float(true_positive_1)/float(true_positive_1+false_positive_1+small_value)
    r_1 = float(true_positive_1)/float(true_positive_1+false_negative_1+small_value)
    # print("标签1的预测情况：", true_positive_1, false_positive_1, false_negative_1, p_1, r_1)
    f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + small_value)
    p_2 = float(true_positive_2)/float(true_positive_2+false_positive_2+small_value)
    r_2 = float(true_positive_2)/float(true_positive_2+false_negative_2+small_value)
    # print("标签2的预测情况：", true_positive_2, false_positive_2, false_negative_2, p_2, r_2)
    f_2 = 2 * p_2 * r_2 / (p_2 + r_2 + small_value)
    p_3 = float(true_positive_3)/float(true_positive_3+false_positive_3+small_value)
    r_3 = float(true_positive_3)/float(true_positive_3+false_negative_3+small_value)
    # print("标签3的预测情况：", true_positive_3, false_positive_3, false_negative_3, p_3, r_3)
    f_3 = 2 * p_3 * r_3 / (p_3 + r_3 + small_value)
    return f_0, f_1, f_2, f_3


def logits_to_predictions(logits_all, column_name, label_to_index):
    # 基于默认置信度获取预测值
    predictions_all = []
    for i in range(len(logits_all)):
        logits_list = logits_all[i]
        label_predict = np.argmax(logits_list)
        predictions_all.append(label_predict)
    # 自主搜索最优参数用于调整置信度
    # automatic_search(logits_all, column_name, label_to_index, test_data_path)
    # 调整置信度后再获取预测值
    # predictions_all = adjust_confidence(logits_all, column_name)
    return predictions_all

if __name__ == '__main__':
    predict()
