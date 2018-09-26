import pandas as pd
import jieba


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents, tokenize_style):
    string_segs = []
    if tokenize_style == "word":
        for content in contents:
            segs = jieba.cut(content)
            string_segs.append(" ".join(segs))
    else:
        for content in contents:
            string_segs.append(" ".join(list(content)))
    return string_segs
