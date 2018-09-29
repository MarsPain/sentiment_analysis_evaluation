import tensorflow as tf
import numpy as np
# from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
from TextCNN_code import config
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from TextCNN_code.data_utils import seg_words, create_dict, get_label_pert, get_labal_weight,\
    shuffle_padding, sentence_word_to_index, get_vector_tfidf
from TextCNN_code.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict

FLAGS = tf.app.flags.FLAGS
# 文件路径参数
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("pkl_dir", "pkl", "dir for save pkl file")
tf.app.flags.DEFINE_string("config_file", "config", "dir for save pkl file")
tf.app.flags.DEFINE_string("tfidf_path", "tfidf.txt", "file for tfidf value dict")
tf.app.flags.DEFINE_string("train_data_path", "../data/sentiment_analysis_trainingset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("dev_data_path", "../data/sentiment_analysis_validationset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("test_data_path", "../data/sentiment_analysis_testa.csv", "path of traning data.")
tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec_word_model.txt", "word2vec's embedding for word")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec_char_model.txt", "word2vec's embedding for char")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/wiki_100.utf8", "word2vec's embedding for char")
# 模型参数
tf.app.flags.DEFINE_integer("num_epochs", config.num_epochs, "number of epochs to run.")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_string("tokenize_style", config.tokenize_style, "tokenize sentence in char,word,or pinyin.default is char")
tf.app.flags.DEFINE_integer("embed_size", config.embed_size, "embedding size")
tf.app.flags.DEFINE_integer("num_filters", config.num_filters, "number of filters")  # 64
tf.app.flags.DEFINE_integer("sentence_len", config.sentence_len, "max sentence length. length should be divide by 3,""which is used by k max pooling.")
tf.app.flags.DEFINE_integer("top_k", config.top_k, "value of top k for k-max polling")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "learning rate")  # 0.001
tf.app.flags.DEFINE_boolean("decay_lr_flag", True, "whether manally decay lr")
tf.app.flags.DEFINE_float("clip_gradients", config.clip_gradients, "clip_gradients")
tf.app.flags.DEFINE_integer("validate_every", config.validate_every, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_float("dropout_keep_prob", config.dropout_keep_prob, "dropout keep probability")
filter_sizes = [2, 3, 4]


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.model_name = None  # 保存模型的文件夹
        self.train_data_df = None   # 训练集
        self.validate_data_df = None    # 验证集
        self.string_train = None    # 训练集的评论字符串
        self.string_valid = None    # 训练集的评论字符串
        self.columns = None  # 列索引的名称
        self.label_train_dict = None  # 用一个字典保存各个评价对象的标签列表
        self.label_valid_dict = None
        self.word_to_index = None   # word到index的映射字典
        self.index_to_word = None   # index到字符word的映射字典
        self.label_to_index = None   # label到index的映射字典
        self.index_to_label = None  # index到label的映射字典
        self.vocab_size = None  # 字符的词典大小
        self.num_classes = None  # 类别标签数量
        self.vectorizer_tfidf = None  # tfidf模型vectorizer_tfidf
        self.label_weight_dict = None   # 存储标签权重

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mn', '--model_name', type=str, nargs='?', help='the name of model')
        args = parser.parse_args()
        self.model_name = args.model_name
        if not self.model_name:
            self.model_name = FLAGS.ckpt_dir
        if not os.path.isdir(self.model_name):   # 创建存储临时字典数据的目录
            os.makedirs(self.model_name)

    def load_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(FLAGS.train_data_path)
        self.validate_data_df = load_data_from_csv(FLAGS.dev_data_path)
        content_train = self.train_data_df.iloc[:100, 1]
        content_valid = self.validate_data_df.iloc[:100, 1]
        logger.info("start seg train data")
        self.string_train = seg_words(content_train, FLAGS.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
        self.string_valid = seg_words(content_valid, FLAGS.tokenize_style)
        # print(self.string_train[0])
        logger.info("complete seg train data")
        self.columns = self.train_data_df.columns.values.tolist()
        # print(self.columns)
        logger.info("load label data")
        self.label_train_dict = {}
        for column in self.columns[2:]:
            label_train = list(self.train_data_df[column].iloc[:])
            self.label_train_dict[column] = label_train
        self.label_valid_dict = {}
        for column in self.columns[2:]:
            label_valid = list(self.validate_data_df[column].iloc[:])
            self.label_valid_dict[column] = label_valid
        # print(self.label_list["location_traffic_convenience"][0], type(self.label_list["location_traffic_convenience"][0]))

    def get_dict(self):
        if not os.path.isdir(FLAGS.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(FLAGS.pkl_dir)
        word_label_dict = os.path.join(FLAGS.pkl_dir, "word_label_dict.pkl")    # 存储word和label与index之间的双向映射字典
        if os.path.exists(word_label_dict):  # 若word_label_path已存在
            with open(word_label_dict, 'rb') as dict_f:
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = pickle.load(dict_f)
        else:   # 重新读取训练数据并创建各个映射字典
            self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = \
                create_dict(self.string_train, self.label_train_dict, word_label_dict)
        print(len(self.word_to_index), self.word_to_index)
        self.vocab_size = len(self.word_to_index)
        # print(self.vocab_size)
        self.num_classes = len(self.label_to_index)
        # print(self.num_classes)

    def get_data(self):
        train_valid_test = os.path.join(FLAGS.pkl_dir, "train_valid_test.pkl")
        if os.path.exists(train_valid_test):    # 若train_valid_test已被处理和存储
            with open(train_valid_test, 'rb') as data_f:
                train_data, valid_data, self.label_weight_dict = pickle.load(data_f)
        else:   # 读取数据集并创建训练集、验证集
            pass
            # 获取tfidf值并存储为tfidf字典
            if not os.path.exists(FLAGS.tfidf_path):
                get_tfidf_and_save(self.string_train, FLAGS.tfidf_path)
            tfidf_dict = load_tfidf_dict(FLAGS.tfidf_path)
            # 根据tfidf_dict获取训练集和验证集的tfidf值向量作为额外的特征向量
            train_vector_tfidf = get_vector_tfidf(self.string_train, tfidf_dict)
            valid_vector_tfidf = get_vector_tfidf(self.string_valid, tfidf_dict)
            # print(train_vector_tfidf[0])
            # 从训练集中获取label_pert_dict（存储标签比例）label_weight_dict（存储标签权重）
            label_pert_dict = get_label_pert(self.train_data_df, self.columns)
            self.label_weight_dict = get_labal_weight(label_pert_dict)
            # 语句序列化，将句子中的word映射成index，作为模型输入
            sentences_train = sentence_word_to_index(self.string_train, self.word_to_index)
            sentences_valid = sentence_word_to_index(self.string_valid, self.word_to_index)
            # 打乱数据、padding,并对评论序列、标签字典、特征向量打包
            train_data = shuffle_padding(sentences_train, self.label_train_dict, train_vector_tfidf)
            valid_data = shuffle_padding(sentences_valid, self.label_valid_dict, valid_vector_tfidf)
            with open(train_valid_test, "wb") as f:
                pickle.dump([train_data, valid_data, self.label_weight_dict], f)
            # 得到batch生成器

if __name__ == "__main__":
    main = Main()
    main.get_parser()
    main.load_data()
    main.get_dict()
    main.get_data()
