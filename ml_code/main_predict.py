#!/user/bin/env python
# -*- coding:utf-8 -*-

from data_utils import seg_words, load_data_from_csv
import logging
import config
import argparse
from sklearn.externals import joblib
from main_train import model_name_file

test_data_path = "../data/sentiment_analysis_testa.csv"
test_data_predict_out_path = "result.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)


def get_parer():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model_name', type=str, nargs='?',
                        help='the name of model')
    args = parser.parse_args()
    model_name = args.model_name
    if not model_name:
        model_name = "model_dict.pkl"
    return model_name


def get_data():
    # load data
    logger.info("start load data")
    test_data_df = load_data_from_csv(test_data_path)
    # seg words
    logger.info("start seg test data")
    content_test = test_data_df.iloc[:, 1]
    content_test = seg_words(content_test)
    logger.info("complete seg test data")
    return test_data_df, content_test


def predict():
    model_name = model_name_file
    # model_name = get_parer()
    test_data_df, content_test = get_data()
    # load model
    logger.info("start load model")
    classifier_dict = joblib.load(config.model_save_path + "_" + model_name)
    columns = test_data_df.columns.tolist()
    # model predict
    logger.info("start predict test data")
    for column in columns[2:]:
        test_data_df[column] = classifier_dict[column].predict(content_test)
        logger.info("compete %s predict" % column)
    test_data_df.to_csv(test_data_predict_out_path, encoding="utf_8_sig", index=False)
    logger.info("compete predict test data")

if __name__ == '__main__':
    predict()











