import numpy as np

print(np.__version__)
a = np.array([4, 5, 0, 1, 2, 3, 6, 7, 8, 9, 10, 11])
print(a)
print(type(a))
print(a.shape)
a.sort()
print(a)

b = np.array([[4, 5, 0], [1, 2, 3], [6, 7, 8], [9, 10, 11]])
b.sort(axis=0)
print(b)

b.sort(axis=1)
print(b)    

