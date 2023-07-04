from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = "C:\\Users\\Subiran\\Desktop\\Nuclei dataset\\Stage1_train\\"
TEST_PATH = "C:\\Users\\Subiran\\Desktop\\Nuclei dataset\\Stage1_test\\"

train_ids = next(os.walk(TRAIN_PATH))[1] # creates a tuple where the firste ntry is TRAIN_PATH and second is subfoldersname
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)

print("Resizing training images and masks")
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
  path = TRAIN_PATH + id_
  img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
  img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True)
  X_train[n] = img # Now everytrain image will be similar added to the X_train list whose dimension will be(720*128*128*3)

  # The difficukt part is the masks because there are so many masks images
  mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
  for mask_file in next(os.walk(path + '/masks/'))[2]:
    mask_ = imread(path + '/masks/' + mask_file)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True), axis = -1)
    mask = np.maximum(mask, mask_)

  Y_train[n] =  mask

  np.save("X_train_save",X_train)
  np.save("Y_train_save", Y_train)


print("The size of X_train is ", X_train.shape, "And the dimension of Y_train is", Y_train.shape)



