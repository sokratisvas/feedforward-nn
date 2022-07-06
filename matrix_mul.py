import numpy as np

a = np.array([[6.6, 4.6, 3.4, 8.2]])
b= np.array([[1.3], [5.3], [8.7], [4]])

c = b @ a

print(c.transpose())


