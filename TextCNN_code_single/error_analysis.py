import pandas as pd
import numpy as np
import re
import os
from TextCNN_code_single.utils import load_data_from_csv


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
    pass


if __name__ == "__main__":
    load_data()
    anayasis()
