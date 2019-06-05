import pandas as pd
import numpy as np
import re
import os
from Attention_RCNN.utils import load_data_from_csv


valid_data_path = "../data/sentiment_analysis_validationset.csv"
log_predict_error_dir = "error_log"


def load_data():
    validate_data_df = load_data_from_csv(valid_data_path)
    columns = validate_data_df.columns.tolist()
    column_name = columns[2]
    error_path = os.path.join(log_predict_error_dir, column_name + ".csv")
    error_log_df = load_data_from_csv(error_path)
    return error_log_df


def anayasis(data):
    # print(data.info())
    data_group_label = data.groupby(["正确标签"])
    print(data_group_label.count())
    data_group_label = data.groupby(["正确标签", "错误标签"])
    print(data_group_label.count())
    print(data_group_label.mean())
    # a = data.loc[data["正确标签"] == 3]
    # print(a)


if __name__ == "__main__":
    data_df = load_data()
    anayasis(data_df)
