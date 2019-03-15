import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png')
gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gimg,127,255,0)
## Contours ##
#Contours can be explained simply as a curve joining all the continuous points, having same color.
#cv2.findContours(srcImg,counter retrieval mode,counter approximation)
#here in output contours is a python list of all contours in the img
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#To draw contours
#cv2.drawContours(srcImg, python list of contours, index of contours, color, thickness)
#if index = -1, it will draw all the contours
img = cv2.drawContours(img, contours, -1, (0,255,0), 3)

## Features of Contours ##
cnt = contours[0]

#1.Moments(centre of mass)
M = cv2.moments(cnt)
#2.Contour Area
area = cv2.contourArea(cnt)
#3.Contour Perimeter
#cv2.arcLength(countour, if True: clossed contour else False: arc)
perimeter = cv2.arcLength(cnt,True)
#4.Contour Approximation (to approximate the shape)
#epsilon: maximum distance from contour to approximated contour.
epsilon = 0.1*cv2.arcLength(cnt,True) #10%
approx_contour = cv2.approxPolyDP(cnt,epsilon,True)
