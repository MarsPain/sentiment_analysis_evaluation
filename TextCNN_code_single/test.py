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

# l = [1, 2, 3]
# print("、".join(str(num) for num in l))

# import cmath
# print(cmath.sqrt(8627200))

# l = [j+round(float(0.1*i), 2) for i in range(1, 10) for j in range(3)]
# print(l)
# l = []
# for i in range(3):
#     for j in range(1, 10):
#         l.append(i + round(0.1 * j, 2))
# print(l)

# from sklearn.feature_extraction.text import TfidfVectorizer
# s = ["我知道 这个 真的 很 好 吃", "这个一点都不 好 吃"]
# vectorizer_tfidf = TfidfVectorizer(analyzer="word", token_pattern=u"(?u)\\b\\w+\\b")
# vectorizer_tfidf.fit(s)
# train_vector_tfidf = vectorizer_tfidf.transform(s)
# word_dict = vectorizer_tfidf.vocabulary_
# print(word_dict)
# print(train_vector_tfidf[0].toarray())

# import tensorflow as tf
# import numpy as np
# a = np.asarray([[1, 1, 1], [1, 1, 1]])
# b = np.asarray([[1, 1, 1], [2, 2, 2]])
# c = a * b
# print(c)
# a = tf.constant([[[1, 1, 1], [2, 2, 2]]])
# b = tf.constant([[[1], [2]]])
# with tf.Session() as sess:
#     b_new = tf.tile(b, [1, 1, 3])
#     # print(b_new)
#     b_new_array = b_new.eval(session=sess)
#     print(b_new_array)
#     c = a * b_new
#     print(c, c.eval(session=sess))
# a = np.asarray([1, 2])
# a = np.reshape(a, [-1, 1])
# print(a)
# print(a.tolist())

# 测试嵌套数组的两种排序方法
# l = [[1, 2], [5, 3], [3, 4]]
# l_2 = sorted(l, key=lambda x: x[0], reverse=True)
# print(l_2)

# l = [[1, 2], [5, 3], [3, 4]]
# l.sort(key=lambda x: x[0], reverse=True)
# print(l)
