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
    corpus = []  # 用于生成tfidf值的语料库
    for i, row in enumerate(data):
        str1 = " ".join(jieba.cut(row[1]))
        str2 = " ".join(jieba.cut(row[2]))
        corpus.append(str1)
        corpus.append(str2)
    # print("string_list:", string_list)
    tfidfvectorizer = TfidfVectorizer()
    tfidf = tfidfvectorizer.fit_transform(corpus)
    # print(tfidf)
    tfidf = tfidfvectorizer.idf_
    # print(tfidf)
    word_to_tfidf = dict(zip(tfidfvectorizer.get_feature_names(), tfidf))
    # print(word_to_tfidf)
    with open(tfidf_path, "w", encoding="utf-8") as f:
        for word, tfidf_score in word_to_tfidf.items():
            # print(k)
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


def load_vector(path):
    """
    加载词向量
    :param path:预训练的词向量文件路径
    :return:word-词向量的映射字典
    """
    vector_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and 'word2vec' in path:
                continue
            word_vec = line.strip().split()
            vec_list = [float(x) for x in word_vec[1:]]
            vector_dict[word_vec[0]] = np.asarray(vec_list)
    # print("vector_dict:", vector_dict)
    return vector_dict
