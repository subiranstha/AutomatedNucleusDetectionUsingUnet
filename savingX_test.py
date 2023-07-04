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

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
print("Resizing testing images")
for n, id_ in tqdm(enumerate(test_ids), total = len(test_ids)):
  path = TEST_PATH + id_
  img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
  img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True)
  X_test[n] = img # Now everytrain image will be similar added to the X_train list whose dimension will be(720*128*128*3)

np.save('X_test_save',  X_test)
print("The dimension of X_test is", X_test.shape)