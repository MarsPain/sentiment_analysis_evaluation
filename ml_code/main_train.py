#!/user/bin/env python
# -*- coding:utf-8 -*-

from ml_code.data_utils import load_data_from_csv, seg_words, get_tfidf_and_save, load_tfidf_dict,\
    get_vector_tfidf, padding_data
from ml_code.model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_code import config
import logging
import numpy as np
from sklearn.externals import joblib
import os
import argparse

train_data_path = "../data/sentiment_analysis_trainingset.csv"
validate_data_path = "../data/sentiment_analysis_validationset.csv"
models_dir = "models"
tfidf_path = "data/tfidf.txt"
max_len = 500

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.models_dir = models_dir  # 模型名称（用于保存模型时的命名）
        self.train_data_df = None   # 训练集
        self.validate_data_df = None    # 验证集
        self.string_train = None  # 训练集的评论（string）
        self.string_valid = None    # 验证集的评论（string）
        self.columns = None  # 列索引名称
        self.vectorizer_tfidf = None    # 特征提取器
        self.classifier_dict = None  # 存储所有训练得到的分类器（每种情绪分别训练一个分类器）
        self.train_data = None  # 用于训练的评论序列
        self.valid_data = None  # 用于验证的评论序列

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mn', '--model_name', type=str, nargs='?', help='the name of model')
        args = parser.parse_args()
        self.models_dir = args.model_name
        if not self.models_dir:
            self.models_dir = models_dir

    def get_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(train_data_path)
        self.validate_data_df = load_data_from_csv(validate_data_path)
        content_train = self.train_data_df.iloc[:2000, 1]    # 训练集评论语句
        logger.info("start seg train data")
        if not os.path.exists("pkl/string_train.pkl"):
            self.string_train = seg_words(content_train)
            joblib.dump(self.string_train, "pkl/string_train.pkl")
        else:
            self.string_train = joblib.load("pkl/string_train.pkl")
        logger.info("complete seg train data")
        content_validate = self.validate_data_df.iloc[:, 1]  # 验证集评论语句
        logger.info("start seg validate data")
        if not os.path.exists("pkl/string_valid.pkl"):
            self.string_valid = seg_words(content_validate)
            joblib.dump(self.string_valid, "pkl/string_valid.pkl")
        else:
            self.string_valid = joblib.load("pkl/string_valid.pkl")
        logger.info("complete seg validate data")
        self.columns = self.train_data_df.columns.values.tolist()   # 获取列索引的名称

    def get_feature(self):
        # logger.info("start train feature extraction")
        # self.vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
        # self.vectorizer_tfidf.fit(self.string_train)
        # logger.info("complete train feature extraction models")
        if not os.path.exists(tfidf_path):
            get_tfidf_and_save(self.string_train, tfidf_path)
        tfidf_dict = load_tfidf_dict(tfidf_path)
        if not os.path.exists("pkl/train_data.pkl"):
            train_vector_tfidf = get_vector_tfidf(self.string_train, tfidf_dict)
            self.train_data = padding_data(train_vector_tfidf, max_len)
            joblib.dump(self.train_data, "pkl/train_data.pkl")
        else:
            self.train_data = joblib.load("pkl/train_data.pkl")
        if not os.path.exists("pkl/valid_data.pkl"):
            valid_vector_tfidf = get_vector_tfidf(self.string_valid, tfidf_dict)
            self.valid_data = padding_data(valid_vector_tfidf, max_len)
            joblib.dump(self.valid_data, "pkl/valid_data.pkl")
        else:
            self.valid_data = joblib.load("pkl/valid_data.pkl")

    def train(self):
        logger.info("start train model")
        self.classifier_dict = dict()
        f1_score_dict = dict()
        for index, column in enumerate(self.columns[2:3]):
            label_train = self.train_data_df[column].iloc[:2000]
            text_classifier = TextClassifier(vectorizer=self.vectorizer_tfidf)
            logger.info("start train %s model" % column)
            text_classifier.fit(self.train_data, label_train)
            logger.info("complete train the %s th model: %s" % (str(index), column))
            f1_score = self.valid(text_classifier, column)
            f1_score_dict[column] = f1_score
            self.save_model(text_classifier, column)
            logger.info("complete save the %s th model: %s" % (str(index), column))
            self.classifier_dict[column] = text_classifier
        logger.info("complete train model")
        self.valid_all(f1_score_dict)

    def valid(self, text_classifier, model_name):
        label_validate = self.validate_data_df[model_name]
        f1_score = text_classifier.get_f1_score(self.valid_data, label_validate)
        return f1_score

    def valid_all(self, f1_score_dict):
        f1_score = np.mean(list(f1_score_dict.values()))
        str_score = "\n"
        for column in self.columns[2:3]:
            str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"
        logger.info("f1_scores: %s\n" % str_score)
        logger.info("f1_score: %s" % f1_score)
        logger.info("complete validate model")

    def save_model(self, text_classifier, model_name):
        logger.info("start save %s model" % models_dir)
        model_save_path = os.path.join(self.models_dir, model_name + ".pkl")
        joblib.dump(text_classifier, model_save_path)

if __name__ == '__main__':
    main = Main()
    # main.get_parser()
    main.get_data()
    main.get_feature()
    main.train()
