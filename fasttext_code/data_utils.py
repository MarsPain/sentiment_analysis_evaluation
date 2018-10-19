import re
import jieba
import numpy as np


def seg_words(contents, tokenize_style):
    string_segs = []
    if tokenize_style == "word":
        stopwords_set = set()
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            segs = jieba.cut(content.strip())
            segs_new = []
            for word in segs:
                if word not in stopwords_set:
                    segs_new.append(word)
                else:
                    pass
            string_segs.append(" ".join(segs_new))
    else:
        for content in contents:
            content = re.sub(" ", "，", content.strip())
            # print(content)
            content = re.sub("\n", "", content.strip())
            # print(" ".join(list(content.strip())))
            string_segs.append(" ".join(list(content.strip())))
    return string_segs


def test_f_score_in_valid_data(valid_label_list, prediction_all):
    f_0, f_1, f_2, f_3 = compute_confuse_matrix(valid_label_list, prediction_all)
    print("f_1, f_0, f_-1, f_-2:", f_0, f_1, f_2, f_3)
    print("f1_score:", (f_0 + f_1 + f_2 + f_3) / 4)
    return (f_0 + f_1 + f_2 + f_3) / 4


def compute_confuse_matrix(labels_all, predictions_all):
    small_value = 0.00001
    length = len(labels_all)
    confuse_matrix_list = [[0, 0, 0] for i in range(4)]
    for i in range(length):
        label_to_index = {1: 0, 0: 1, -1: 2, -2: 3}
        label = label_to_index[labels_all[i]]
        predict = label_to_index[predictions_all[i]]
        if label == predict:
            confuse_matrix_list[label][0] += 1
        else:
            confuse_matrix_list[predict][1] += 1
            confuse_matrix_list[label][2] += 1
    f_score_list = []
    for i in range(4):
        p = float(confuse_matrix_list[i][0])/float(confuse_matrix_list[i][0]+confuse_matrix_list[i][1]+small_value)
        r = float(confuse_matrix_list[i][0])/float(confuse_matrix_list[i][0]+confuse_matrix_list[i][2]+small_value)
        f = 2 * p * r / (p + r + small_value)
        print("标签%s的预测情况：" % str(i))
        print(confuse_matrix_list[i][0], confuse_matrix_list[i][0], confuse_matrix_list[i][1], p, r, f)
        f_score_list.append(f)
    return f_score_list
