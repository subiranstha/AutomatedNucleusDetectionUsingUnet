import numpy as np
import matplotlib.pyplot as plt

arr = np.load('abc.npy')
print(arr.shape)

img1 = arr[0]
plt.imshow(img1)
plt.show()

