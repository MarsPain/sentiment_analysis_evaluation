import tensorflow as tf
import numpy as np
# from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
from TextCNN_code_multi import config
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from TextCNN_code_multi.data_utils import seg_words, create_dict, shuffle_padding, sentence_word_to_index,\
    get_vector_tfidf, BatchManager, get_max_len, get_weights_for_current_batch, compute_confuse_matrix,\
    get_labal_weight
from TextCNN_code_multi.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding
from TextCNN_code_multi.model import TextCNN

FLAGS = tf.app.flags.FLAGS
# 文件路径参数
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("pkl_dir", "pkl", "dir for save pkl file")
tf.app.flags.DEFINE_string("config_file", "config", "dir for save pkl file")
tf.app.flags.DEFINE_string("tfidf_path", "./data/tfidf.txt", "file for tfidf value dict")
tf.app.flags.DEFINE_string("train_data_path", "../data/sentiment_analysis_trainingset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("dev_data_path", "../data/sentiment_analysis_validationset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("test_data_path", "../data/sentiment_analysis_testa.csv", "path of traning data.")
tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec_word_model.txt", "word2vec's embedding for word")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/wiki_100.utf8", "word2vec's embedding for word")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec_char_model.txt", "word2vec's embedding for char")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/wiki_100.utf8", "word2vec's embedding for char")
# 模型参数
tf.app.flags.DEFINE_integer("num_classes", config.num_classes, "number of label class.")
tf.app.flags.DEFINE_integer("num_epochs", config.num_epochs, "number of epochs to run.")
tf.app.flags.DEFINE_integer("batch_size", config.batch_size, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_string("tokenize_style", config.tokenize_style, "tokenize sentence in char,word,or pinyin.default is char")
tf.app.flags.DEFINE_integer("vocab_size", config.vocab_size, "size of vocab")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding", True, "whether to use embedding or not")
tf.app.flags.DEFINE_integer("embed_size", config.embed_size, "embedding size")
tf.app.flags.DEFINE_integer("num_filters", config.num_filters, "number of filters")  # 64
tf.app.flags.DEFINE_integer("max_len", config.max_len, "max sentence length. length should be divide by 3,""which is used by k max pooling.")
tf.app.flags.DEFINE_integer("top_k", config.top_k, "value of top k for k-max polling")
tf.app.flags.DEFINE_float("learning_rate", config.learning_rate, "learning rate")  # 0.001
tf.app.flags.DEFINE_boolean("decay_lr_flag", True, "whether manally decay lr")
tf.app.flags.DEFINE_float("clip_gradients", config.clip_gradients, "clip_gradients")
tf.app.flags.DEFINE_integer("validate_every", config.validate_every, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_float("dropout_keep_prob", config.dropout_keep_prob, "dropout keep probability")
filter_sizes = config.filter_sizes


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
        content_train = self.train_data_df.iloc[:, 1]
        content_valid = self.validate_data_df.iloc[:, 1]
        logger.info("start seg train data")
        self.string_train = seg_words(content_train, FLAGS.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
        print("训练集大小：", len(self.string_train))
        self.string_valid = seg_words(content_valid, FLAGS.tokenize_style)
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
        logger.info("start get dict")
        if not os.path.isdir(FLAGS.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(FLAGS.pkl_dir)
        word_label_dict = os.path.join(FLAGS.pkl_dir, "word_label_dict.pkl")    # 存储word和label与index之间的双向映射字典
        if os.path.exists(word_label_dict):  # 若word_label_path已存在
            with open(word_label_dict, 'rb') as dict_f:
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = pickle.load(dict_f)
        else:   # 重新读取训练数据并创建各个映射字典
            self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = \
                create_dict(self.string_train, self.label_train_dict, word_label_dict, FLAGS.vocab_size)
        # print(len(self.word_to_index), self.word_to_index)
        logger.info("complete get dict")

    def get_data(self):
        logger.info("start get data")
        train_valid_test = os.path.join(FLAGS.pkl_dir, "train_valid_test.pkl")
        if os.path.exists(train_valid_test):    # 若train_valid_test已被处理和存储
            with open(train_valid_test, 'rb') as data_f:
                train_data, valid_data, self.label_weight_dict = pickle.load(data_f)
        else:   # 读取数据集并创建训练集、验证集
            # 获取tfidf值并存储为tfidf字典
            if not os.path.exists(FLAGS.tfidf_path):
                get_tfidf_and_save(self.string_train, FLAGS.tfidf_path)
            tfidf_dict = load_tfidf_dict(FLAGS.tfidf_path)
            # 根据tfidf_dict获取训练集和验证集的tfidf值向量作为额外的特征向量
            train_vector_tfidf = get_vector_tfidf(self.string_train, tfidf_dict)
            valid_vector_tfidf = get_vector_tfidf(self.string_valid, tfidf_dict)
            # print(train_vector_tfidf[0])
            # 语句序列化，将句子中的word和label映射成index，作为模型输入
            sentences_train, self.label_train_dict = sentence_word_to_index(self.string_train, self.word_to_index, self.label_train_dict, self.label_to_index)
            sentences_valid, self.label_valid_dict = sentence_word_to_index(self.string_valid, self.word_to_index, self.label_valid_dict, self.label_to_index)
            # print(sentences_train[0])
            # print(self.label_train_dict["location_traffic_convenience"])
            # 打乱数据、padding,并对评论序列、特征向量、标签字典打包
            # max_sentence = get_max_len(sentences_train)  # 获取最大评论序列长度
            train_data = shuffle_padding(sentences_train, train_vector_tfidf, self.label_train_dict, FLAGS.max_len)
            valid_data = shuffle_padding(sentences_valid, valid_vector_tfidf, self.label_valid_dict, FLAGS.max_len)
            # 从训练集中获取label_weight_dict（存储标签权重）
            self.label_weight_dict = get_labal_weight(train_data[2], self.columns, config.num_classes)
            with open(train_valid_test, "wb") as f:
                pickle.dump([train_data, valid_data, self.label_weight_dict], f)
        # self.label_weight_dict = get_labal_weight(train_data[2], self.columns, config.num_classes)
        print("训练集大小：", len(train_data[0]), "验证集大小：", len(valid_data[0]))
        # 获取train、valid数据的batch生成类
        self.train_batch_manager = BatchManager(train_data, int(FLAGS.batch_size))
        print("训练集批次数量：", self.train_batch_manager.len_data)
        self.valid_batch_manager = BatchManager(valid_data, int(FLAGS.batch_size))
        logger.info("complete get data")

    def train_control(self):
        """
        控制针对每种评价对象分别进行训练、验证和模型保存，所有模型保存的文件夹都保存在总文件夹ckpt中
        模型文件夹以评价对象进行命名
        :return:
        """
        # 开始多任务模型的构建
        logger.info("start train")
        column_name_list = self.columns
        column_name_mini_list = column_name_list[2:5]   # 选择评价对象
        logger.info("start %s model train" % " ".join(column_name_mini_list))
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            self.train(sess, column_name_mini_list)
        logger.info("complete %s model train" % " ".join(column_name_mini_list))
        logger.info("complete all models' train")

    def train(self, sess, column_name_mini_list):
        num_task = len(column_name_mini_list)
        text_cnn, saver = self.create_model(sess, "location")
        curr_epoch = sess.run(text_cnn.epoch_step)
        iteration = 0
        best_acc = 0.50
        best_f1_score = 0.20
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss_list = [0 for i in range(num_task)]
            eval_acc_list = [0 for i in range(num_task)]
            loss_all = 0
            eval_acc_all = 0
            counter = 0
            # train
            for batch in self.train_batch_manager.iter_batch(shuffle=True):
                iteration += 1
                input_x, features_vector, input_y_dict = batch
                input_y_list = []
                weights_list = []
                for column_name in column_name_mini_list:
                    input_y = input_y_dict[column_name]
                    input_y_list.append(input_y)
                    weights = get_weights_for_current_batch(input_y, self.label_weight_dict[column_name])   # 根据类别权重参数更新训练集各标签的权重
                    weights_list.append(weights)
                # print("input_y:", input_y)
                feed_dict = {text_cnn.input_x: input_x, text_cnn.features_vector: features_vector, text_cnn.input_y_list: input_y_list,
                             text_cnn.weights_list: weights_list, text_cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             text_cnn.iter: iteration}
                curr_loss_list, curr_acc_list, lr, _ = sess.run([text_cnn.loss_val_list, text_cnn.accuracy_list, text_cnn.learning_rate, text_cnn.train_op], feed_dict)
                counter += 1
                for i in range(num_task):
                    loss_list[i] += curr_loss_list[i]
                    eval_acc_list[i] += curr_acc_list[i]
                    loss_all += curr_loss_list[i]
                    eval_acc_all += curr_acc_list[i]
                if counter % 100 == 0:  # steps_check
                    for index, column_name in enumerate(column_name_mini_list):
                        print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f\tModel %s" % (epoch, counter, loss_list[index]/float(counter), eval_acc_list[index]/float(counter), lr, column_name))
                    print("Average： Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f" % (epoch, counter, loss_all/float(counter*num_task), eval_acc_all/float(counter*num_task), lr))
            print("going to increment epoch counter....")
            sess.run(text_cnn.epoch_increment)
            # valid
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_accc, f1_scoree, weights_label = self.evaluate(sess, text_cnn, self.valid_batch_manager, iteration, column_name_mini_list, num_task, epoch)
                print("【Average Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f" % (epoch, eval_loss, eval_accc, f1_scoree))
                # save model to checkpoint
                if f1_scoree > best_f1_score:
                    save_path = FLAGS.ckpt_dir + "/location" + "/model.ckpt"
                    print("going to save model. eval_f1_score:", f1_scoree, ";previous best f1 score:", best_f1_score,
                          ";eval_acc", str(eval_accc), ";previous best_acc:", str(best_acc))
                    saver.save(sess, save_path, global_step=epoch)
                    best_acc = eval_accc
                    best_f1_score = f1_scoree
                if FLAGS.decay_lr_flag and (epoch != 0 and (epoch == 10 or epoch == 20 or epoch == 24 or epoch == 28)):
                    for i in range(1):  # decay learning rate if necessary.
                        print(i, "Going to decay learning rate by half.")
                        sess.run(text_cnn.learning_rate_decay_half_op)

    def create_model(self, sess, column_name):
        text_cnn = TextCNN()
        saver = tf.train.Saver()
        model_save_dir = FLAGS.ckpt_dir + "/" + column_name
        if os.path.exists(model_save_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))
            if FLAGS.decay_lr_flag:
                for i in range(2):  # decay learning rate if necessary.
                    print(i, "Going to decay learning rate by half.")
                    sess.run(text_cnn.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            if FLAGS.use_pretrained_embedding:  # 加载预训练的词向量
                print("===>>>going to use pretrained word embeddings...")
                old_emb_matrix = sess.run(text_cnn.Embedding.read_value())
                new_emb_matrix = load_word_embedding(old_emb_matrix, FLAGS.word2vec_model_path, FLAGS.embed_size, self.index_to_word)
                word_embedding = tf.constant(new_emb_matrix, dtype=tf.float32)  # 转为tensor
                t_assign_embedding = tf.assign(text_cnn.Embedding, word_embedding)  # 将word_embedding复制给text_cnn.Embedding
                sess.run(t_assign_embedding)
                print("using pre-trained word emebedding.ended...")
        return text_cnn, saver

    def evaluate(self, sess, text_cnn, batch_manager, iteration, column_name_mini_list, num_task, epoch):
        small_value = 0.00001
        # file_object = open('data/log_predict_error.txt', 'a')
        eval_loss_all, eval_accc_all, eval_counter = 0.0, 0.0, 0
        true_positive_0_all, false_positive_0_all, false_negative_0_all, true_positive_1_all, false_positive_1_all, false_negative_1_all,\
        true_positive_2_all, false_positive_2_all, false_negative_2_all, true_positive_3_all, false_positive_3_all, false_negative_3_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        eval_loss_list = [0 for i in range(num_task)]
        accc_list = [0 for i in range(num_task)]
        weights_label = {}  # weight_label[label_index]=(number,correct)
        confuse_matrix_list = [[[0, 0, 0] for j in range(4)] for i in range(num_task)]
        for batch in batch_manager.iter_batch(shuffle=True):
            eval_x, features_vector, eval_y_dict = batch
            eval_y_list = []
            weights_list = []
            for column_name in column_name_mini_list:
                eval_y = eval_y_dict[column_name]
                eval_y_list.append(eval_y)
                weights = get_weights_for_current_batch(eval_y, self.label_weight_dict[column_name])  # 根据类别权重参数更新训练集各标签的权重
                weights_list.append(weights)
            feed_dict = {text_cnn.input_x: eval_x, text_cnn.features_vector: features_vector, text_cnn.input_y_list: eval_y_list,
                         text_cnn.weights_list: weights_list, text_cnn.dropout_keep_prob: 1.0, text_cnn.iter: iteration}
            curr_eval_loss_list, curr_accc_list, logits_list = sess.run([text_cnn.loss_val_list, text_cnn.accuracy_list, text_cnn.logits_list], feed_dict)
            for i in range(num_task):
                eval_loss_list[i] += curr_eval_loss_list[i]
                accc_list[i] += curr_accc_list[i]
                eval_loss_all += curr_eval_loss_list[i]
                eval_accc_all += curr_accc_list[i]
            for index, column_name in enumerate(column_name_mini_list):
                true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1,\
                true_positive_2, false_positive_2, false_negative_2, true_positive_3, false_positive_3, false_negative_3 = compute_confuse_matrix(logits_list[index], eval_y_list[index], small_value)
                confuse_matrix_list[index][0][0] += true_positive_0
                confuse_matrix_list[index][0][1] += false_positive_0
                confuse_matrix_list[index][0][2] += false_negative_0
                confuse_matrix_list[index][1][0] += true_positive_1
                confuse_matrix_list[index][1][1] += false_positive_1
                confuse_matrix_list[index][1][2] += false_negative_1
                confuse_matrix_list[index][2][0] += true_positive_2
                confuse_matrix_list[index][2][1] += false_positive_2
                confuse_matrix_list[index][2][2] += false_negative_2
                confuse_matrix_list[index][3][0] += true_positive_3
                confuse_matrix_list[index][3][1] += false_positive_3
                confuse_matrix_list[index][3][2] += false_negative_3
            eval_counter += 1
            # write_predict_error_to_file(file_object, logits, eval_y, self.index_to_word, eval_x1, eval_x2)    # 获取被错误分类的样本（后期再处理）
        # print("标签0的预测情况：", true_positive_0, false_positive_0, false_negative_0)
        f1_score_all = 0
        for index, column_name in enumerate(column_name_mini_list):
            p_0 = float(confuse_matrix_list[index][0][0])/float(confuse_matrix_list[index][0][0]+confuse_matrix_list[index][0][1]+small_value)
            r_0 = float(confuse_matrix_list[index][0][0])/float(confuse_matrix_list[index][0][0]+confuse_matrix_list[index][0][2]+small_value)
            f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + small_value)
            # print("标签1的预测情况：", true_positive_1, false_positive_1, false_negative_1)
            p_1 = float(confuse_matrix_list[index][1][0])/float(confuse_matrix_list[index][1][0] + confuse_matrix_list[index][1][1] + small_value)
            r_1 = float(confuse_matrix_list[index][1][0])/float(confuse_matrix_list[index][1][0] + confuse_matrix_list[index][1][2] + small_value)
            f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + small_value)
            # print("标签2的预测情况：", true_positive_2, false_positive_2, false_negative_2)
            p_2 = float(confuse_matrix_list[index][2][0])/float(confuse_matrix_list[index][2][0] + confuse_matrix_list[index][2][1] + small_value)
            r_2 = float(confuse_matrix_list[index][2][0])/float(confuse_matrix_list[index][2][0] + confuse_matrix_list[index][2][2] + small_value)
            f_2 = 2 * p_2 * r_2 / (p_2 + r_2 + small_value)
            # print("标签3的预测情况：", true_positive_3, false_positive_3, false_negative_3)
            p_3 = float(confuse_matrix_list[index][3][0])/float(confuse_matrix_list[index][3][0] + confuse_matrix_list[index][3][1] + small_value)
            r_3 = float(confuse_matrix_list[index][3][0])/float(confuse_matrix_list[index][3][0] + confuse_matrix_list[index][3][2] + small_value)
            f_3 = 2 * p_3 * r_3 / (p_3 + r_3 + small_value)
            f1_score = (f_0 + f_1 + f_2 + f_3) / 4
            print("【Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f\tf_0:%.3f\tf_1:%.3f\tf_2:%.3f\tf_3:%.3f\tModel: %s" %
                  (epoch, eval_loss_list[index]/float(eval_counter), accc_list[index]/float(eval_counter), f1_score, f_0, f_1, f_2, f_3, column_name))
            f1_score_all += f1_score
        return eval_loss_all/float(eval_counter*num_task), eval_accc_all/float(eval_counter*num_task), f1_score_all/num_task, weights_label

if __name__ == "__main__":
    main = Main()
    main.get_parser()
    main.load_data()
    main.get_dict()
    main.get_data()
    main.train_control()
