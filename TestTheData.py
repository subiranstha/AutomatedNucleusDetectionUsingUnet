import tensorflow as tf
from keras.models import load_model
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np

model = load_model('mymodel1_scratch.h5')
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

PATH ="C:\\Users\\Subiran\\Desktop\\Nuclei dataset\\Stage1_test\\0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732\\images\\"
name = "0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png"

img_path = PATH + name
img = imread(img_path)[:, :, :IMG_CHANNELS]
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
print(img.shape)
print("The input image data is")
# Actually the image can only be in the form of float if with its value between [0,1] and later multiplying by 255
# If the image is in the integer format then its value has to be in between 0, 255 and if there is some floating values in between then it gotta be change
img = np.uint8(img)
plt.imshow(img)
plt.show()

#print(model.summary())

img = np.expand_dims(img, axis=0)
print(img.shape)
output = model.predict(img)
print("The shape of output", output.shape)
#print(output[0])
x = output[0]
x[x>=0.5] =255
x[x<0.5] = 0
plt.imshow(x)
plt.show()