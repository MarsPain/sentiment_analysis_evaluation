import numpy as np
import re
import pandas as pd
import tensorflow as tf


# s = "sadasdas  dsadas d dasd dsa  dsa"
# print(s.split(" "))

# l = [1, 2, 3, 4]
# print(sum(l))

# 会自动忽略不匹配的列，然后不匹配的列在array中以list形式存在
# a = np.asarray([[1,2,3],[4,5,6],[7,8,9,10]])
# print(a.shape)
# print(a)
# a = np.asarray([[1,2,3],[4,5,6],[7,8,10]])
# print(a.shape)
# a = np.asarray([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[10]]])
# print(a.shape)

# 测试字典的排序
# d = {"a": 5, "b": 4, "我": 7}
# print(d)
# d_sort = sorted(d.items(), key=lambda x: x[1])
# print(d_sort)
# word_list_sort = [v[0] for i, v in enumerate(d_sort)]
# print(word_list_sort)

# df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]])
# print(df)
# l = ["a", "b"]
# df[1] = l
# print(df)

# 测试权重参数
# y = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# value = np.bincount(y)
# print(type(value), value)
# print(10 / (2 * value))

# 测试在字符串的格式化中使用列表的字符串拼接
# print("start %s model train" % " ".join(["a", "b"]))

# 测试对二维tensor和三维tensor进行循环读取
# a_list = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
# a_tensor = tf.convert_to_tensor(a_list)
# with tf.Session() as sess:
#     size_0 = a_tensor.shape[0]
#     print(size_0)
#     for i in range(size_0):
#         print(a_tensor[i])
