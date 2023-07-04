import copy
import cv2
mask_location = 'D:\\New 294 images only\\Fokusdurchmesser_294mm\\A1\\label.png'

img_location = 'D:\\New 294 images only\\Fokusdurchmesser_294mm\\A1\\img.png'
mask = cv2.imread(mask_location)
mask_width, mask_height = mask.shape[1], mask.shape[0]
mask = cv2.resize(mask, (500, 500))

image = cv2.imread(img_location)
#image = cv2.resize(image, (mask_width, mask_height))
image = cv2.resize(image, (500, 500))
#print("The new dimension of the image is", image.shape[1], image.shape[0])

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask[mask>0]= 255
mask[mask==0] = 0
cv2.imshow("Binarized mask", mask)

contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
copy1 = copy.deepcopy(image)
cv2.drawContours(copy1,contours, -1, (0,255,0), 2)
cv2.imshow("Contoured Image", copy1)
cv2.waitKey(0)