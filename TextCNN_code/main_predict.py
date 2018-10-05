from TextCNN_code import config
import logging
from sklearn.externals import joblib
import pickle
import os
import tensorflow as tf
import math
import random
from TextCNN_code.data_utils import seg_words, create_dict, get_label_pert, get_labal_weight,\
    shuffle_padding, get_vector_tfidf, get_max_len,\
    get_weights_for_current_batch, compute_confuse_matrix
from TextCNN_code.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding
from TextCNN_code.model import TextCNN
from TextCNN_code.main_train import Main

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"

test_data_path = "../data/sentiment_analysis_testa.csv"
test_data_predict_out_path = "result.csv"
models_dir = "ckpt"
word_label_dict = "pkl/word_label_dict.pkl"
tfidf_path = "data/tfidf.txt"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def get_data():
    # load data
    logger.info("start load data")
    test_data_df = load_data_from_csv(test_data_path)
    if os.path.exists(test_data_path):    # 若train_valid_test已被处理和存储
        with open(test_data_path, 'rb') as data_f:
            test_data = pickle.load(data_f)
    else:
        # seg words
        logger.info("start seg test data")
        content_test = test_data_df.iloc[:, 1]
        string_test = seg_words(content_test, "word")
        logger.info("complete seg test data")
        with open(word_label_dict, 'rb') as dict_f:
            word_to_index, index_to_word, label_to_index, index_to_label = pickle.load(dict_f)
        tfidf_dict = load_tfidf_dict(tfidf_path)
        test_vector_tfidf = get_vector_tfidf(string_test, tfidf_dict)
        sentences_test = sentence_word_to_index(string_test, word_to_index)
        sentences_padding = padding_data(sentences_test, config.max_len)
        vector_tfidf_padding = padding_data(test_vector_tfidf, config.max_len)
        test_data = [sentences_padding, vector_tfidf_padding]
        with open(test_data_path, "wb") as f:
            pickle.dump([test_data], f)
    test_batch_manager = BatchManager(test_data, int(config.batch_size))
    logger.info("complete load data")
    return test_data_df, test_batch_manager


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


def predict():
    # model_name = get_parer()
    test_data_df, test_batch_manager = get_data()
    columns = test_data_df.columns.tolist()
    # model predict
    logger.info("start predict test data")
    for column in columns[2:3]:
        model_path = os.path.join(models_dir, column)
        main = Main()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            logits_list = []
            for batch in test_batch_manager.iter_batch(shuffle=True):
                test_x, features_vector = batch
                text_cnn, saver = main.create_model(sess, config)
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                logger.info("compete load %s model and start predict" % column)
                feed_dict = {text_cnn.input_x: test_x, text_cnn.features_vector: features_vector,
                             text_cnn.dropout_keep_prob: 1.0}
                logits = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.logits], feed_dict)
                logits_list.append(logits)
            test_data_df[column] = logits_list
        logger.info("compete %s predict" % column)
    logger.info("compete predict test data")

if __name__ == '__main__':
    predict()
