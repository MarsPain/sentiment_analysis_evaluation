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
models_dir = "models"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


class Main:
    def __init__(self):
        self.models_dir = models_dir  # 模型名称（用于保存模型时的命名）
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
        self.models_dir = args.model_name
        if not self.models_dir:
            self.models_dir = models_dir

    def get_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(train_data_path)
        self.validate_data_df = load_data_from_csv(validate_data_path)
        content_train = self.train_data_df.iloc[:, 1]
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
        f1_score_dict = dict()
        for index, column in enumerate(self.columns[2:3]):
            label_train = self.train_data_df[column].iloc[:]
            text_classifier = TextClassifier(vectorizer=self.vectorizer_tfidf)
            logger.info("start train %s model" % column)
            text_classifier.fit(self.string_train, label_train)
            logger.info("complete train the %s th model: %s" % (str(index), column))
            f1_score = self.valid(text_classifier, column)
            f1_score_dict[column] = f1_score
            self.save_model(text_classifier, column)
            logger.info("complete save the %s th model: %s" % (str(index), column))
            self.classifier_dict[column] = text_classifier
        logger.info("complete train model")
        self.valid_all(f1_score_dict)

    def valid(self, text_classifier, model_name):
        content_validate = self.validate_data_df.iloc[:, 1]
        logger.info("start seg validate data")
        content_validate = seg_words(content_validate)
        logger.info("complete seg validate data")
        logger.info("start validate model")
        label_validate = self.validate_data_df[model_name]
        f1_score = text_classifier.get_f1_score(content_validate, label_validate)
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
