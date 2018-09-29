#!/user/bin/env python
# -*- coding:utf-8 -*-

from ml_code.data_utils import load_data_from_csv, seg_words
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
model_name_file = "model_dict.pkl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.model_name = model_name_file # 模型名称（用于保存模型时的命名）
        self.train_data_df = None   # 训练集
        self.validate_data_df = None    # 验证集
        self.string_train = None  # 训练集的评论（string）
        self.columns = None  # 列索引名称
        self.vectorizer_tfidf = None    # 特征提取器
        self.classifier_dict = None  # 存储所有训练得到的分类器（每种情绪分别训练一个分类器）

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-mn', '--model_name', type=str, nargs='?', help='the name of model')
        args = parser.parse_args()
        self.model_name = args.model_name
        if not self.model_name:
            self.model_name = "model_dict.pkl"

    def get_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(train_data_path)
        self.validate_data_df = load_data_from_csv(validate_data_path)
        content_train = self.train_data_df.iloc[:100, 1]
        logger.info("start seg train data")
        self.string_train = seg_words(content_train)
        logger.info("complete seg train data")
        self.columns = self.train_data_df.columns.values.tolist()   # 获取列索引的名称

    def get_feature(self):
        logger.info("start train feature extraction")
        self.vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
        self.vectorizer_tfidf.fit(self.string_train)
        logger.info("complete train feature extraction models")
        # logger.info("vocab shape: %s" % np.shape(self.vectorizer_tfidf.vocabulary_.keys()))

    def train(self):
        logger.info("start train model")
        self.classifier_dict = dict()
        for column in self.columns[2:]:
            label_train = self.train_data_df[column].iloc[:100]
            text_classifier = TextClassifier(vectorizer=self.vectorizer_tfidf)
            logger.info("start train %s model" % column)
            text_classifier.fit(self.string_train, label_train)
            logger.info("complete train %s model" % column)
            self.classifier_dict[column] = text_classifier
        logger.info("complete train model")
        self.valid()
        self.save_model()

    def valid(self):
        content_validate = self.validate_data_df.iloc[:, 1]
        logger.info("start seg validate data")
        content_validate = seg_words(content_validate)
        logger.info("complete seg validate data")
        logger.info("start validate model")
        f1_score_dict = dict()
        for column in self.columns[2:]:
            label_validate = self.validate_data_df[column]
            text_classifier = self.classifier_dict[column]
            f1_score = text_classifier.get_f1_score(content_validate, label_validate)
            f1_score_dict[column] = f1_score
        f1_score = np.mean(list(f1_score_dict.values()))
        str_score = "\n"
        for column in self.columns[2:]:
            str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"
        logger.info("f1_scores: %s\n" % str_score)
        logger.info("f1_score: %s" % f1_score)
        logger.info("complete validate model")

    def save_model(self):
        logger.info("start save model")
        model_save_path = config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        joblib.dump(self.classifier_dict, model_save_path + "_" + self.model_name)
        logger.info("complete save model")


if __name__ == '__main__':
    main = Main()
    # main.get_parser()
    main.get_data()
    main.get_feature()
    main.train()
