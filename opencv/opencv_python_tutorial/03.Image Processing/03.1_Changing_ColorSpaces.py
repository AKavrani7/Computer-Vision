import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png')

## 1.Changing Colorspace ##
## BGR -> GRAY
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
## BGR -> HSV
hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## 2. Blue Object Detection ##

#2.a define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
#2.b Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsvImg, lower_blue, upper_blue)
#2.c Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)
#2.d tried to remove blue content by using HSV method
try_image = img - res

##How to find range of blue color space in HSV
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#print hsv_green
#[[[ 60 255 255]]]

#Now you take [H-10, 100,100] and [H+10, 255, 255] as
#lower bound and upper bound respectively
