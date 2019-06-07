import numpy as np


# a = np.array([True, False], dtype=float)
# print(a)

# a = np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4],[1,2,3,4]])
# print(np.sum(a, axis=0, keepdims=True))
# print(np.sum(a, axis=1, keepdims=True))

# n = np.array([1,2,3,4])
# a, b, c, d = n
# print(a, b, c, d)

# n = [1, 2, 3, 4]
# a, b, c, d = n
# print(a, b, c, d)

a = np.array([0, 0, 0, 0])
for i in range(4):
    a[i] = 54
print(a)
