import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def get_tfidf_and_save(data, tfidf_path):
    """
    获取tfidf值并写入到文件中
    :param data:
    :param tfidf_path:
    :return:
    """
    vectorizer_tfidf = TfidfVectorizer()
    vectorizer_tfidf.fit(data)
    train_vector_tfidf = vectorizer_tfidf.transform(data)
    # print("train_vector_tfidf", train_vector_tfidf[0])
    tfidf_values_list = train_vector_tfidf.toarray()    # 保存所有句子所包含词汇的tfidf值的稀疏矩阵（失去了原有的句子的顺序）
    # print("tfidf_values_list:", len(tfidf_values_list[0]), tfidf_values_list[0], sum(tfidf_values_list[0]))
    word_list = vectorizer_tfidf.get_feature_names()    # 所有被赋tfidf值的单词列表
    # print("word_list:", word_list)
    word_to_tfidf = {}  # 存储word到tfidf的映射字典
    # for i in range(len(tfidf_values_list)):
    for i in range(len(tfidf_values_list)):
        for j in range(len(word_list)):
            tfidf_value = tfidf_values_list[i][j]
            # print(word_list[j], float(tfidf_value))
            if float(tfidf_value) != 0.0:
                # print("YES")
                word_to_tfidf[word_list[j]] = tfidf_value
    with open(tfidf_path, "w", encoding="utf-8") as f:
        for word, tfidf_score in word_to_tfidf.items():
            f.write(word+"|||"+str(tfidf_score)+"\n")


def load_tfidf_dict(tfidf_path):
    """
    加载tfidf值
    :param tfidf_path:word-tfidf的映射字典
    :return:
    """
    tfidf_dict = {}
    with open(tfidf_path, "r", encoding="utf-8") as f:
        for line in f:
            word, tfidf_score = line.strip().split("|||")
            tfidf_dict[word] = float(tfidf_score)
    # print("tfidf_dict:", tfidf_dict)
    return tfidf_dict
