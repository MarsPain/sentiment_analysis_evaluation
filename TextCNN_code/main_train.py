import tensorflow as tf
import numpy as np
# from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
import config
import logging
import os
import argparse
from TextCNN_code.data_utils import load_data_from_csv, seg_words

FLAGS = tf.app.flags.FLAGS
# 文件路径参数
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("pkl_dir", "pkl", "dir for save pkl file")
tf.app.flags.DEFINE_string("config_file", "config", "dir for save pkl file")
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
        self.train_data_df = None
        self.validate_data_df = None
        self.string_train = None    # 训练集的评论字符串
        self.columns = None  # 列索引的名称

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
        logger.info("start seg train data")
        self.string_train = seg_words(content_train, FLAGS.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
        # print(self.string_train[0])
        logger.info("complete seg train data")
        self.columns = self.train_data_df.columns.values.tolist()
        # print(self.columns)

    def get_dict(self):
        pass

if __name__ == "__main__":
    main = Main()
    main.get_parser()
    main.load_data()
