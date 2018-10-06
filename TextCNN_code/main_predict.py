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

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"

# test_data_path = "../data/sentiment_analysis_testa.csv"
test_data_path = "result.csv"
test_data_pkl = "pkl/test_data.pkl"
test_data_predict_out_path = "result.csv"
models_dir = "ckpt"
word_label_dict = "pkl/word_label_dict.pkl"
tfidf_path = "data/tfidf.txt"
word2vec_model_path = "data/word2vec_word_model.txt"

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
        with open(test_data_pkl, "wb") as f:
            pickle.dump([test_data], f)
    print(len(test_data[0]), len(test_data[1]))
    test_batch_manager = BatchManager(test_data, int(config.batch_size))
    logger.info("complete load data")
    return test_data_df, test_batch_manager, index_to_word, index_to_label


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
    print("===>>>going to use pretrained word embeddings...")
    old_emb_matrix = sess.run(text_cnn.Embedding.read_value())
    new_emb_matrix = load_word_embedding(old_emb_matrix, word2vec_model_path, config.embed_size, index_to_word)
    word_embedding = tf.constant(new_emb_matrix, dtype=tf.float32)  # 转为tensor
    t_assign_embedding = tf.assign(text_cnn.Embedding, word_embedding)  # 将word_embedding复制给text_cnn.Embedding
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")
    return text_cnn, saver


def predict():
    # model_name = get_parer()
    test_data_df, test_batch_manager, index_to_word, index_to_label = get_data()
    columns = test_data_df.columns.tolist()
    # model predict
    logger.info("start predict test data")
    for column in columns[21:22]:
        model_path = os.path.join(models_dir, column)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            predictions_all = []
            text_cnn, saver = create_model(sess, index_to_word)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            logger.info("compete load %s model and start predict" % column)
            for batch in test_batch_manager.iter_batch(shuffle=True):
                test_x, features_vector = batch
                # print(len(test_x[0]), test_x[0])
                feed_dict = {text_cnn.input_x: test_x, text_cnn.features_vector: features_vector,
                             text_cnn.dropout_keep_prob: 1.0}
                predictions = sess.run([text_cnn.predictions], feed_dict)
                predictions_list = list(predictions[0])
                # print(predictions_list)
                predictions_all.extend(predictions_list)
            # print(len(predictions_all))
            # 将predictions映射到label，预测得到的是index。
            logger.info("start transfer index to label")
            for i in range(len(predictions_all)):
                predictions_all[i] = index_to_label[predictions_all[i]]
            # print(predictions_all)
            test_data_df[column] = predictions_all
        logger.info("compete %s predict" % column)
    test_data_df.to_csv(test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")

if __name__ == '__main__':
    predict()
