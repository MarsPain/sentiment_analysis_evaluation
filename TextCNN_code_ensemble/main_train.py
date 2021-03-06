import tensorflow as tf
import numpy as np
# from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
from TextCNN_code_ensemble import config
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from TextCNN_code_ensemble.data_utils import seg_words, create_dict, shuffle_padding, sentence_word_to_index,\
    get_vector_tfidf, BatchManager, get_max_len, get_weights_for_current_batch, compute_confuse_matrix,\
    get_labal_weight, get_least_label, afresh_sampling, get_sample_weights, get_weights_for_current_batch_and_sample,\
    get_f_scores_all
from TextCNN_code_ensemble.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding
from TextCNN_code_ensemble.model import TextCNN

FLAGS = tf.app.flags.FLAGS
# 文件路径参数
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("pkl_dir", "pkl", "dir for save pkl file")
tf.app.flags.DEFINE_string("config_file", "config", "dir for save pkl file")
tf.app.flags.DEFINE_string("tfidf_path", "./data/tfidf.txt", "file for tfidf value dict")
tf.app.flags.DEFINE_string("train_data_path", "../data/sentiment_analysis_trainingset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("dev_data_path", "../data/sentiment_analysis_validationset.csv", "path of traning data.")
tf.app.flags.DEFINE_string("test_data_path", "../data/sentiment_analysis_testa.csv", "path of traning data.")
tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec_word_model_sg.txt", "word2vec's embedding for word")
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
        # self.train_batch_manager = None  # train数据batch生成类
        self.valid_batch_manager = None  # valid数据batch生成类
        self.least_label_dict = None    # 获取每种评价对象的标签中数量最少的标签数量
        self.train_data = None  # 被打包好的训练数据

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
        if not os.path.isdir(FLAGS.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(FLAGS.pkl_dir)
        string_train_valid = os.path.join(FLAGS.pkl_dir, "string_train_valid.pkl")
        if os.path.exists(string_train_valid):  # 若word_label_path已存在
            with open(string_train_valid, 'rb') as f:
                self.string_train, self.string_valid = pickle.load(f)
        else:
            self.string_train = seg_words(content_train, FLAGS.tokenize_style)  # 根据tokenize_style对评论字符串进行分词
            self.string_valid = seg_words(content_valid, FLAGS.tokenize_style)
            with open(string_train_valid, 'wb') as f:
                pickle.dump([self.string_train, self.string_valid], f)
        print("训练集大小：", len(self.string_train))
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
                self.train_data, valid_data, self.label_weight_dict, self.least_label_dict = pickle.load(data_f)
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
            self.train_data = shuffle_padding(sentences_train, train_vector_tfidf, self.label_train_dict, FLAGS.max_len)
            valid_data = shuffle_padding(sentences_valid, valid_vector_tfidf, self.label_valid_dict, FLAGS.max_len)
            # 从训练集中获取label_weight_dict（存储标签权重）
            self.label_weight_dict = get_labal_weight(self.train_data[2], self.columns, config.num_classes)
            self.least_label_dict = get_least_label(self.train_data[2], self.columns)
            with open(train_valid_test, "wb") as f:
                pickle.dump([self.train_data, valid_data, self.label_weight_dict, self.least_label_dict], f)
        print("训练集大小：", len(self.train_data[0]), "验证集大小：", len(valid_data[0]))
        # 获取train、valid数据的batch生成类
        # self.train_batch_manager = BatchManager(self.train_data, int(FLAGS.batch_size))
        # print("训练集批次数量：", self.train_batch_manager.len_data)
        self.valid_batch_manager = BatchManager(valid_data, int(FLAGS.batch_size))
        logger.info("complete get data")

    def train_control(self):
        """
        控制针对每种评价对象分别进行训练、验证和模型保存，所有模型保存的文件夹都保存在总文件夹ckpt中
        模型文件夹以评价对象进行命名
        :return:
        """
        logger.info("start train")
        column_name_list = self.columns
        column_name = column_name_list[config.column_index]   # 选择评价对象
        logger.info("start %s model train" % column_name)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            self.train(sess, column_name)
        logger.info("complete %s model train" % column_name)
        logger.info("complete all models' train")

    def train(self, sess, column_name):
        for model_index in range(config.num_models):
            sess.run(tf.global_variables_initializer())
            print("%s 的第 %s 个模型" % (column_name, str(model_index)))
            train_batch_sample_manager = afresh_sampling(self.train_data, self.least_label_dict, column_name, int(FLAGS.batch_size))
            text_cnn, saver = self.create_model(sess, column_name, model_index)
            curr_epoch = sess.run(text_cnn.epoch_step)
            iteration = 0
            best_acc = 0.50
            best_f1_score = 0.20
            batch_num = train_batch_sample_manager.len_data
            print("训练集批次数量：", batch_num)
            sample_weights_list = []
            for batch in train_batch_sample_manager.iter_batch(shuffle=False):
                input_x, features_vector, input_y = batch
                sample_weights_list.append([1 for i in range(len(input_x))])
            for epoch in range(curr_epoch, FLAGS.num_epochs):
                loss, eval_acc, counter = 0.0, 0.0, 0
                sample_weights_list_new = []
                # train
                input_y_all = []
                predictions_all = []
                for batch in train_batch_sample_manager.iter_batch(shuffle=False):
                    iteration += 1
                    input_x, features_vector, input_y = batch
                    input_y_all.extend(input_y)
                    # print("input_y:", input_y)
                    index = iteration % batch_num - 1 if iteration % batch_num != 0 else batch_num - 1
                    sample_weights_mini_list = sample_weights_list[index]
                    weights = get_weights_for_current_batch_and_sample(input_y, self.label_weight_dict[column_name], sample_weights_mini_list)   # 根据类别权重参数更新训练集各标签的权重
                    feed_dict = {text_cnn.input_x: input_x, text_cnn.features_vector: features_vector, text_cnn.input_y: input_y,
                                 text_cnn.weights: weights, text_cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                 text_cnn.iter: iteration}
                    curr_loss, curr_acc, lr, _, predictions = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.learning_rate, text_cnn.train_op, text_cnn.predictions],
                                                                       feed_dict)
                    predictions_all.extend(predictions)
                    loss, eval_acc, counter = loss+curr_loss, eval_acc+curr_acc, counter+1
                    # predictions = list(predictions[0])
                    # print("predictions:", predictions, len(predictions))
                    sample_weights_mini_list_new = get_sample_weights(input_y, predictions, sample_weights_mini_list)
                    sample_weights_list_new.append(sample_weights_mini_list_new)
                    if counter % 10 == 0:  # steps_check
                        print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f" % (epoch, counter, loss/float(counter), eval_acc/float(counter), lr))
                sample_weights_list = sample_weights_list_new
                f_0, f_1, f_2, f_3 = get_f_scores_all(predictions_all, input_y_all, 0.00001)  # test_f_score_in_valid_data
                print("f_0, f_1, f_2, f_3:", f_0, f_1, f_2, f_3)
                print("f1_score:", (f_0 + f_1 + f_2 + f_3) / 4)
                print("going to increment epoch counter....")
                sess.run(text_cnn.epoch_increment)
                # valid
                if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_accc, f1_scoree, f_0, f_1, f_2, f_3, weights_label = self.evaluate(sess, text_cnn, self.valid_batch_manager, iteration, column_name)
                    print("【Validation】Epoch %d\t f_0:%.3f\tf_1:%.3f\tf_2:%.3f\tf_3:%.3f" % (epoch, f_0, f_1, f_2, f_3))
                    print("【Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f" % (epoch, eval_loss, eval_accc, f1_scoree))
                    # save model to checkpoint
                    if f1_scoree > best_f1_score:
                        save_path = FLAGS.ckpt_dir + "/" + column_name + "/" + str(model_index) + "/model.ckpt"
                        print("going to save model. eval_f1_score:", f1_scoree, ";previous best f1 score:", best_f1_score,
                              ";eval_acc", str(eval_accc), ";previous best_acc:", str(best_acc))
                        saver.save(sess, save_path)
                        best_acc = eval_accc
                        best_f1_score = f1_scoree
                    if FLAGS.decay_lr_flag and (epoch != 0 and (epoch == 5 or epoch == 8 or epoch == 11)):
                        for i in range(1):  # decay learning rate if necessary.
                            print(i, "Going to decay learning rate by half.")
                            sess.run(text_cnn.learning_rate_decay_half_op)

    def create_model(self, sess, column_name, model_index):
        text_cnn = TextCNN()
        saver = tf.train.Saver()
        model_save_dir = FLAGS.ckpt_dir + "/" + column_name + "/" + str(model_index)
        if os.path.exists(model_save_dir):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))
            if False:
                for i in range(1):  # decay learning rate if necessary.
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

    def evaluate(self, sess, text_cnn, batch_manager, iteration, column_name):
        small_value = 0.00001
        # file_object = open('data/log_predict_error.txt', 'a')
        eval_loss, eval_accc, eval_counter = 0.0, 0.0, 0
        true_positive_0_all, false_positive_0_all, false_negative_0_all, true_positive_1_all, false_positive_1_all, false_negative_1_all,\
        true_positive_2_all, false_positive_2_all, false_negative_2_all, true_positive_3_all, false_positive_3_all, false_negative_3_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        weights_label = {}  # weight_label[label_index]=(number,correct)
        for batch in batch_manager.iter_batch(shuffle=True):
            eval_x, features_vector, eval_y_dict = batch
            eval_y = eval_y_dict[column_name]
            weights = get_weights_for_current_batch(eval_y, self.label_weight_dict[column_name])   # 根据类别权重参数更新训练集各标签的权重
            feed_dict = {text_cnn.input_x: eval_x, text_cnn.features_vector: features_vector, text_cnn.input_y: eval_y,
                         text_cnn.weights: weights, text_cnn.dropout_keep_prob: 1.0, text_cnn.iter: iteration}
            curr_eval_loss, curr_accc, logits = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.logits], feed_dict)
            true_positive_0, false_positive_0, false_negative_0, true_positive_1, false_positive_1, false_negative_1,\
            true_positive_2, false_positive_2, false_negative_2, true_positive_3, false_positive_3, false_negative_3 = compute_confuse_matrix(logits, eval_y, small_value)
            true_positive_0_all += true_positive_0
            false_positive_0_all += false_positive_0
            false_negative_0_all += false_negative_0
            true_positive_1_all += true_positive_1
            false_positive_1_all += false_positive_1
            false_negative_1_all += false_negative_1
            true_positive_2_all += true_positive_2
            false_positive_2_all += false_positive_2
            false_negative_2_all += false_negative_2
            true_positive_3_all += true_positive_3
            false_positive_3_all += false_positive_3
            false_negative_3_all += false_negative_3
            # write_predict_error_to_file(file_object, logits, eval_y, self.index_to_word, eval_x1, eval_x2)    # 获取被错误分类的样本（后期再处理）
            eval_loss, eval_accc, eval_counter = eval_loss+curr_eval_loss, eval_accc+curr_accc, eval_counter+1
        # print("标签0的预测情况：", true_positive_0, false_positive_0, false_negative_0)
        p_0 = float(true_positive_0_all)/float(true_positive_0_all+false_positive_0_all+small_value)
        r_0 = float(true_positive_0_all)/float(true_positive_0_all+false_negative_0_all+small_value)
        f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + small_value)
        # print("标签1的预测情况：", true_positive_1, false_positive_1, false_negative_1)
        p_1 = float(true_positive_1_all)/float(true_positive_1_all+false_positive_1_all+small_value)
        r_1 = float(true_positive_1_all)/float(true_positive_1_all+false_negative_1_all+small_value)
        f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + small_value)
        # print("标签2的预测情况：", true_positive_2, false_positive_2, false_negative_2)
        p_2 = float(true_positive_2_all)/float(true_positive_2_all+false_positive_2_all+small_value)
        r_2 = float(true_positive_2_all)/float(true_positive_2_all+false_negative_2_all+small_value)
        f_2 = 2 * p_2 * r_2 / (p_2 + r_2 + small_value)
        # print("标签3的预测情况：", true_positive_3, false_positive_3, false_negative_3)
        p_3 = float(true_positive_3_all)/float(true_positive_3_all+false_positive_3_all+small_value)
        r_3 = float(true_positive_3_all)/float(true_positive_3_all+false_negative_3_all+small_value)
        f_3 = 2 * p_3 * r_3 / (p_3 + r_3 + small_value)
        f1_score = (f_0 + f_1 + f_2 + f_3) / 4
        return eval_loss/float(eval_counter), eval_accc/float(eval_counter), f1_score, f_0, f_1, f_2, f_3, weights_label

if __name__ == "__main__":
    main = Main()
    main.get_parser()
    main.load_data()
    main.get_dict()
    main.get_data()
    main.train_control()
