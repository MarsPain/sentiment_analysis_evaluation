import pandas as pd
import numpy as np
from Attention_RCNN.utils import load_data_from_csv


def automatic_search(logits_all, column_name, label_to_index, valid_data_path):
    max_f1 = 0
    if column_name == "location_traffic_convenience":
        best_param = [0, 0, 0, 0]
        param_list_1 = []
        for i in range(0, 3):
            for j in range(1, 10):
                param_list_1.append(i + round(0.1 * j, 2))
        param_list_2 = []
        for i in range(0, 3):
            for j in range(1, 10):
                param_list_2.append(i + round(0.1 * j, 2))
        param_list_3 = []
        for i in range(0, 6):
            for j in range(1, 10):
                param_list_3.append(i + round(0.1 * j, 2))
        for param_1 in param_list_1:
            for param_2 in param_list_2:
                for param_3 in param_list_3:
                    predictions_all = []
                    for i in range(len(logits_all)):
                        logits_list = logits_all[i]
                        label_predict = np.argmax(logits_list)
                        if (logits_list[1]-logits_list[3]) < param_1 and (logits_list[0]+logits_list[2]) < param_2 and label_predict == 1:   # 减少将标签3错误地识别为1的数量
                            label_predict = 3
                        if (logits_list[0]-logits_list[2] < param_3) and label_predict == 0:  # 减少将标签1错误地识别为0的数量
                            label_predict = 1
                        predictions_all.append(label_predict)
                    f1_score = test_f_score_in_valid_data(predictions_all, column_name, label_to_index, valid_data_path)
                    if max_f1 < f1_score:
                        max_f1 = f1_score
                        best_param = [param_1, param_2, param_3]
                        print("max_f1:", max_f1)
                        print("best_param:", " ".join(str(param) for param in best_param))
        print("max_f1:", max_f1)
        print("best_param:", " ".join(str(param) for param in best_param))


def test_f_score_in_valid_data(predictions_all, column_name, label_to_index, valid_data_path):
    validate_data_df = load_data_from_csv(valid_data_path)
    label_valid = list(validate_data_df[column_name].iloc[:])
    for i in range(len(predictions_all)):
        label_valid[i] = label_to_index[label_valid[i]]
    # print("predictions_all:", len(predictions_all), predictions_all)
    # print("label_valid:", len(label_valid), label_valid)
    f_0, f_1, f_2, f_3 = compute_confuse_matrix(predictions_all, label_valid, 0.00001)
    print("f_0, f_1, f_2, f_3:", f_0, f_1, f_2, f_3)
    print("f1_score:", (f_0 + f_1 + f_2 + f_3) / 4)
    return (f_0 + f_1 + f_2 + f_3) / 4


def compute_confuse_matrix(predictions_all, label, small_value):
    length = len(label)
    true_positive_0 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_0 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_0 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_1 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_1 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_1 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_2 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_2 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_2 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    true_positive_3 = 0  # TP:if label is true('0'), and predict is true('0')
    false_positive_3 = 0  # FP:if label is false('1,2,3'),but predict is ture('0')
    false_negative_3 = 0  # FN:if label is true('0'),but predict is false('1,2,3')
    for i in range(length):
        # 用于计算0的精确率和召回率
        if label[i] == 0 and predictions_all[i] == 0:
            true_positive_0 += 1
        elif (label[i] == 1 or label[i] == 2 or label[i] == 3) and predictions_all[i] == 0:
            false_positive_0 += 1
        elif label[i] == 0 and (predictions_all[i] == 1 or predictions_all[i] == 2 or predictions_all == 3):
            false_negative_0 += 1
        # 用于计算1的精确率和召回率
        if label[i] == 1 and predictions_all[i] == 1:
            true_positive_1 += 1
        elif (label[i] == 0 or label[i] == 2 or label[i] == 3) and predictions_all[i] == 1:
            false_positive_1 += 1
        elif label[i] == 1 and (predictions_all[i] == 0 or predictions_all[i] == 2 or predictions_all[i] == 3):
            false_negative_1 += 1
        # 用于计算2的精确率和召回率
        if label[i] == 2 and predictions_all[i] == 2:
            true_positive_2 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 3) and predictions_all[i] == 2:
            false_positive_2 += 1
        elif label[i] == 2 and (predictions_all[i] == 0 or predictions_all[i] == 1 or predictions_all[i] == 3):
            false_negative_2 += 1
        # 用于计算3的精确率和召回率
        if label[i] == 3 and predictions_all[i] == 3:
            true_positive_3 += 1
        elif (label[i] == 0 or label[i] == 1 or label[i] == 2) and predictions_all[i] == 3:
            false_positive_3 += 1
        elif label[i] == 3 and (predictions_all[i] == 0 or predictions_all[i] == 1 or predictions_all[i] == 2):
            false_negative_3 += 1
    p_0 = float(true_positive_0)/float(true_positive_0+false_positive_0+small_value)
    r_0 = float(true_positive_0)/float(true_positive_0+false_negative_0+small_value)
    # print("标签0的预测情况：", true_positive_0, false_positive_0, false_negative_0, p_0, r_0)
    f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + small_value)
    p_1 = float(true_positive_1)/float(true_positive_1+false_positive_1+small_value)
    r_1 = float(true_positive_1)/float(true_positive_1+false_negative_1+small_value)
    # print("标签1的预测情况：", true_positive_1, false_positive_1, false_negative_1, p_1, r_1)
    f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + small_value)
    p_2 = float(true_positive_2)/float(true_positive_2+false_positive_2+small_value)
    r_2 = float(true_positive_2)/float(true_positive_2+false_negative_2+small_value)
    # print("标签2的预测情况：", true_positive_2, false_positive_2, false_negative_2, p_2, r_2)
    f_2 = 2 * p_2 * r_2 / (p_2 + r_2 + small_value)
    p_3 = float(true_positive_3)/float(true_positive_3+false_positive_3+small_value)
    r_3 = float(true_positive_3)/float(true_positive_3+false_negative_3+small_value)
    # print("标签3的预测情况：", true_positive_3, false_positive_3, false_negative_3, p_3, r_3)
    f_3 = 2 * p_3 * r_3 / (p_3 + r_3 + small_value)
    return f_0, f_1, f_2, f_3


def adjust_confidence(logits_all, column_name):
    predictions_all = []
    # 调整置信度后再获取预测值
    # location_traffic_convenience的置信度调整
    if column_name == "location_traffic_convenience":
        param_1 = 1.2
        param_2 = 0.1
        param_3 = 4.3
        predictions_all = []
        for i in range(len(logits_all)):
            logits_list = logits_all[i]
            label_predict = np.argmax(logits_list)
            if (logits_list[1]-logits_list[3]) < param_1 and (logits_list[0]+logits_list[2]) < param_2 and label_predict == 1:   # 减少将标签3错误地识别为1的数量
                label_predict = 3
            if (logits_list[0]-logits_list[2] < param_3) and label_predict == 0:  # 减少将标签1错误地识别为0的数量
                label_predict = 1
            predictions_all.append(label_predict)
    if column_name == "location_distance_from_business_district":
        param_1 = 1.2
        param_2 = 1.1
        param_3 = 4.3
        predictions_all = []
        for i in range(len(logits_all)):
            logits_list = logits_all[i]
            label_predict = np.argmax(logits_list)
            if (logits_list[1]-logits_list[3]) < param_1 and (logits_list[0]+logits_list[2]) < param_2 and label_predict == 1:   # 减少将标签3错误地识别为1的数量
                label_predict = 3
            if (logits_list[0]-logits_list[2] < param_3) and label_predict == 0:  # 减少将标签1错误地识别为0的数量
                label_predict = 1
            predictions_all.append(label_predict)
    return predictions_all
