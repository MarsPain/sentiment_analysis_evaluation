import pandas as pd
import jieba
import codecs
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def get_tfidf_and_save(data, tfidf_path, tokenize_style):
    """
    获取tfidf值并写入到文件中
    :param data:
    :param tfidf_path:
    :param tokenize_style:
    :return:
    """
    if tokenize_style == "word":
        vectorizer_tfidf = TfidfVectorizer()
    else:
        vectorizer_tfidf = TfidfVectorizer(analyzer="char")
    vectorizer_tfidf.fit(data)
    word_dict = vectorizer_tfidf.vocabulary_
    word_dict_sorted = sorted(word_dict.items(), key=lambda x: x[1])
    word_list_sort = [v[0] for i, v in enumerate(word_dict_sorted)]
    word_list_sort_dict = {}
    for index, word in enumerate(word_list_sort):
        word_list_sort_dict[word] = index
    with open(tfidf_path, "wb") as f:
        pickle.dump([vectorizer_tfidf, word_list_sort_dict], f)
    return vectorizer_tfidf, word_list_sort_dict


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


def load_word_embedding(emb_matrix, word2vec_model_path, embed_size, index_to_word):
        print("Loading pretrained embeddings from", word2vec_model_path)
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(codecs.open(word2vec_model_path, 'r', 'utf-8')):
            line = line.rstrip().split()
            if len(line) == embed_size + 1:
                pre_trained[line[0]] = np.asarray([float(x) for x in line[1:]]).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(index_to_word)
        for i in range(n_words):
            word = index_to_word[i]
            if word in pre_trained:
                emb_matrix[i] = pre_trained[word]
                c_found += 1
            elif word.lower() in pre_trained:
                emb_matrix[i] = pre_trained[word.lower()]
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pre_trained:
                emb_matrix[i] = pre_trained[
                    re.sub('\d', '0', word.lower())
                ]
                c_zeros += 1
        print('Loaded %i pretrained embeddings.' % len(pre_trained))
        print('%i / %i (%.4f%%) words have been initialized with ''pretrained embeddings.' %
              (c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))
        print('%i found directly, %i after lowercasing, ''%i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
        return emb_matrix
