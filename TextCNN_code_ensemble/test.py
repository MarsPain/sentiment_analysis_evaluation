import numpy as np
import re
import pandas as pd


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

# l = [1 for i in range(10)]
# for i in range(1, 20):
#     print((i-1)%4, i%4)
#     print(l[((i-1)%4)*3: (i%4)*3])
# l = [1, 2, 3]
# print(l[(2)])
