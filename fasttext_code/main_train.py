import pandas as pd
import numpy as np
import os
import csv
from fastText import train_supervised, load_model
import json
import pickle
import logging
from fasttext_code import config
from fasttext_code.utils import load_data_from_csv, write_to_txt
from fasttext_code.data_utils import seg_words, test_f_score_in_valid_data


train_data_path = "../data/sentiment_analysis_trainingset.csv"
dev_data_path = "../data/sentiment_analysis_validationset.csv"
pkl_dir = "pkl"
word_vector_dir = "word_vector"


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
        self.label_weight_dict = None   # 存储标签权重
        self.train_batch_manager = None  # train数据batch生成类
        self.valid_batch_manager = None  # valid数据batch生成类

    def load_data(self):
        logger.info("start load data")
        self.train_data_df = load_data_from_csv(train_data_path)
        self.validate_data_df = load_data_from_csv(dev_data_path)
        content_train = self.train_data_df.iloc[:, 1]
        content_valid = self.validate_data_df.iloc[:, 1]
        logger.info("start seg train data")
        if not os.path.isdir(pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(pkl_dir)
        string_train_valid = os.path.join(pkl_dir, "string_train_valid.pkl")
        if os.path.exists(string_train_valid):  # 若word_label_path已存在
            with open(string_train_valid, 'rb') as f:
                self.string_train, self.string_valid = pickle.load(f)
        else:
            self.string_train = seg_words(content_train, config.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
            self.string_valid = seg_words(content_valid, config.tokenize_style)
            with open(string_train_valid, 'wb') as f:
                pickle.dump([self.string_train, self.string_valid], f)
        print("训练集大小：", len(self.string_train))
        logger.info("complete seg train data")
        self.columns = self.train_data_df.columns.values.tolist()
        logger.info("load label data")
        self.label_train_dict = {}
        for column in self.columns[2:]:
            label_train = list(self.train_data_df[column].iloc[:])
            self.label_train_dict[column] = label_train
        self.label_valid_dict = {}
        for column in self.columns[2:]:
            label_valid = list(self.validate_data_df[column].iloc[:])
            self.label_valid_dict[column] = label_valid

    def data_to_txt(self):
        columns_list = self.columns
        for column_name in columns_list[2:]:
            train_txt_path = os.path.join("data", column_name + "_train.txt")
            valid_txt_path = os.path.join("data", column_name + "_val.txt")
            train_label_list = self.label_train_dict[column_name]
            valid_label_list = self.label_valid_dict[column_name]
            write_to_txt(self.string_train, train_label_list, train_txt_path)
            write_to_txt(self.string_valid, valid_label_list, valid_txt_path)

    def train(self):
        column_name = self.columns[config.column_index]
        train_txt_path = os.path.join("data", column_name + "_train.txt")
        valid_txt_path = os.path.join("data", column_name + "_val.txt")
        model_path = os.path.join("ckpt", column_name + "_model")
        if not os.path.exists(model_path):
            classifier = train_supervised(train_txt_path)
            classifier.save_model(model_path)
        else:
            classifier = load_model(model_path)

    def evaluate(self):
        # 基于预测值计算F值
        column_name = self.columns[config.column_index]
        model_path = os.path.join("ckpt", column_name + "_model")
        string_valid_all = self.string_valid    # 验证集的评论语句
        valid_label_list = self.label_valid_dict[column_name]   # 该评价对象的标签列表
        classifier = load_model(model_path)
        prediction_all = []
        len_val_data = len(string_valid_all)
        for i in range(len_val_data):
            string = self.string_valid[i]
            predict_tuple = classifier.predict(string)
            # print(list(predict_tuple[0])[0])
            predict = int(list(predict_tuple[0])[0][9:])
            # print(predict)
            prediction_all.append(predict)
        test_f_score_in_valid_data(valid_label_list, prediction_all)

    # 结合get_word_vector方法和get_words方法获取最终训练得到的词向量
    def get_word_vector(self):
        column_name = self.columns[config.column_index]
        model_path = os.path.join("ckpt", column_name + "_model")
        word_vector_path = os.path.join(word_vector_dir, column_name + "_fasttext.txt")
        classifier = load_model(model_path)
        words_list = classifier.get_words()
        print("词汇总数", len(words_list))
        num_words = len(words_list)
        with open(word_vector_path, "w", encoding="utf-8") as f:
            f.write(str(num_words) + " " + str(100) + "\n")
            for i in range(num_words):
                word = words_list[i]
                word_vector = classifier.get_word_vector(word)
                # print("word_vector:", type(word_vector), len(word_vector))
                f.write(word + " " + " ".join([str(vector_float) for vector_float in word_vector]) + "\n")

if __name__ == "__main__":
    main = Main()
    main.load_data()
    # main.data_to_txt()
    main.train()
    # main.evaluate()
    main.get_word_vector()
