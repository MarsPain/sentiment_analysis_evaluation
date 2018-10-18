import pandas as pd


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def write_to_txt(string, label_list, txt_path):
    len_data = len(string)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(len_data):
            f.write(string[i] + " " + "__label__" + str(label_list[i]) + "\n")
