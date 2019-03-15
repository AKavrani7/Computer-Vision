# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
### Feature Detection(Finding the image features) ###

## 1.Harris Corner Detection##
#It basically finds the difference in intensity for a displacement of (𝑢, 𝑣) in all directions.
#E(u,v) = summiation{ w(x,y) * [I(x + u, y + v)−I(x,y)]^2 }
#w(x,y): window function (gives weights to pixels underneath)
#I(x+u,y+v): shifted intensity
#I(x,y): Intensity

## 𝑅 = 𝜆1*𝜆2 − 𝑘(𝜆1 + 𝜆2)^2

#cv2.cornerHarris(gray_img,blocksize,ksize,k)
#gray_img - Input image, it should be grayscale and float32 type
harris = img
harris_gray = gray
harris_gray = np.float32(harris_gray)
#blockSize - It is the size of neighbourhood considered for corner detection
#ksize - Aperture parameter of Sobel derivative used.
#k - Harris detector free parameter
dst = cv2.cornerHarris(harris_gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
harris[dst>0.01*dst.max()]=[0,0,255] #Red Color

## 2.Shi-Tomasi Corner Detector & Good Features to Track
#𝑅 = 𝑚𝑖𝑛(𝜆1, 𝜆2)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
