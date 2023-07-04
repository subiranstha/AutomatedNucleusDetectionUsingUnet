import os
from tqdm import tqdm
from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import savetxt

TRAIN_PATH = "C:\\Users\\Subiran\\Desktop\\Nuclei dataset\\Stage1_train\\"

train_ids = next(os.walk(TRAIN_PATH))[1]
print(train_ids)
print("The length is", len(train_ids))

img_path = TRAIN_PATH + train_ids[0] + "\\images\\" + train_ids[0] + ".png"
img_path2 = TRAIN_PATH + train_ids[1] + "\\images\\" + train_ids[1] + ".png"
X_train = np.zeros((2, 128, 128, 3),dtype = np.uint8)
img = imread(img_path)[:,:,:3]
img = resize(img, (128,128), mode = 'constant', preserve_range=True)

X_train[0] = img
X_train[0] = X_train[0].astype(np.uint8)
img2 = imread(img_path2)[:,:,:3]
img2 = resize(img2, (128,128), mode = 'constant', preserve_range=True)
X_train[1] = img
X_train[1] = X_train[1].astype(np.uint8)

#print(X_train)
print("The type is", type(X_train), X_train.shape)
#X_train_2d = X_train.reshape(X_train.shape[0], -1)

np.save('abc', X_train)

#plt.imshow(X_train)
#plt.show()
