# -*- coding: cp1252 -*-
import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png')

## Image Smoothing or blurring with Low Pass Filter ##

#Functioning
#OpenCV provides a filter through cv2.filter2D() which convolve on an image
kernel = np.ones((5,5),np.float32)/25 
Basic_filter = cv2.filter2D(img,-1,kernel)

#LPF helps in removing noise, or blurring the image.
#HPF filters helps in finding edges in an image.

### Blurring Techniques
## 1. Averaging
#convolving the image with a normalized box filter.
#It simply takes the average of all the pixels under kernel area
#and replaces the central element with this average.
Averaging_blur = cv2.blur(img,(5,5)) ## 5X5 kernel size

## 2.Gaussian Filtering
#Instead of a box filter consisting of equal filter coefficients,
#a Gaussian kernel is used.
#cv2.GaussianBlur(img,w_kernel,h_kernel, sigmaX, sigmaY)
#If signaX = sigmaY = 0, they are calculated from the kernel size.
Gaussian_blur = cv2.GaussianBlur(img,(5,5),0) #sigmaX = sigmaY

## 3.Median Filtering
#Under the kernel window, it computes the median of all the pixels
#and the central pixel is replaced with this median value.
#This is highly effective in removing salt-and-pepper noise.
median_blur = cv2.medianBlur(img,5)

## 4.Bilateral Filtering
#cv2.bilateralFilter() is highly effective at noise removal while preserving edges.
#bilateral_filter = Gaussian_filter(space domain) + Gaussian_filter(pixel intensity)
#Gaussian_filter(space domain): only pixels are ‘spatial neighbors’ are considered for filtering
#Gaussian_filter(pixel intensity): ensures that only those pixels with intensities similar to that of
#the central pixel (‘intensity neighbors’) are included to compute the blurred intensity value.
bilaterl_filter_blur = cv2.bilateralFilter(img,9,75,75)
