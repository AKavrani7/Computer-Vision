import cv2
import numpy as np

from matplotlib import pyplot as plt

#### 1.Image Thresholding #####
# if px value > threshold value px value = assign_value(white or black)
# cv2.threshold(source_image(gray), threshold value, maxVal, assign_value)
grayImg = cv2.imread('sudoku.jpg',0)
ret,Simple_th = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)

#### 2.Adaptive Thresholding
#The algorithm calculate the threshold for a small regions of the image.
#So we get different thresholds for different regions of the same image
#and it gives us better results for images with varying illumination.
#Three PARAMETERS:
block_size = 11
constant_c = 2
#A. Adaptive Method - It decides how thresholding value is calculated.
# cv.ADAPTIVE_THRESH_MEAN_C : threshold value is the mean of neighbourhood area.
Adaptive_th_M = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant_c)
# cv.ADAPTIVE_THRESH_GAUSSIAN_C : threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
Adaptive_th_G = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_c)

#B. Block Size - It decides the size of neighbourhood area.
#C. C - It is just a constant which is subtracted from the mean or weighted mean calculated.

#### 3.Otsu's thresholding
#In simple words, bimodal image is an image whose histogram has two peaks).
#For that image, we can approximately take a value in the middle of those peaks as threshold

#For this, our cv2.threshold() function is used, but pass an extra flag,
#cv2.THRESH_OTSU. For threshold value, simply pass zero. Then the algorithm
#finds the optimal threshold value and returns you as the second output.
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
