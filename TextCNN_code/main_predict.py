from TextCNN_code import config
import logging
from sklearn.externals import joblib
from TextCNN_code.data_utils import seg_words, create_dict, get_label_pert, get_labal_weight,\
    shuffle_padding, sentence_word_to_index, get_vector_tfidf, BatchManager, get_max_len,\
    get_weights_for_current_batch, compute_confuse_matrix
from TextCNN_code.utils import load_data_from_csv, get_tfidf_and_save, load_tfidf_dict,\
    load_word_embedding
from TextCNN_code.model import TextCNN

test_data_path = "../data/sentiment_analysis_testa.csv"
test_data_predict_out_path = "result.csv"
models_dir = "ckpt"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def get_data():
    # load data
    logger.info("start load data")
    test_data_df = load_data_from_csv(test_data_path)
    # seg words
    logger.info("start seg test data")
    content_test = test_data_df.iloc[:, 1]
    content_test = seg_words(content_test, "word")
    logger.info("complete seg test data")
    return test_data_df, content_test


def predict():
    # model_name = get_parer()
    test_data_df, content_test = get_data()
    columns = test_data_df.columns.tolist()
    # model predict
    logger.info("start predict test data")
    for column in columns[2:3]:
        text_classifier = joblib.load(models_dir + "/" + column + ".pkl")
        logger.info("compete load %s model" % column)
        test_data_df[column] = text_classifier.predict(content_test)
        logger.info("compete %s predict" % column)
    test_data_df.to_csv(test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")

if __name__ == '__main__':
    predict()
