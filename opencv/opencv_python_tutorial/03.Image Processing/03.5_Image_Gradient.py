import cv2
import numpy as np

gimg = cv2.imread('Naruto_Uzumaki.png',0) ##GrayScale_Image

## Image Gradients or High Pass Filter ##

## 1. Sobel and Scharr Derivatives
#Sobel or Scharr operators is a joint Gausssian smoothing plus differentiation operation.
#cv2.Sobel(src, output_image data type,dx,dy,ksize)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#If ksize = -1, a 3x3 Scharr filter is used.
#Xorder
#Scharr
#[ -1 -10 -3 ]
#[  0  0   0 ]
#[  3  10  3 ]
#SobelX =
#[ -1 0 1 ]
#[ -2 0 2 ]
#[ -1 0 1 ]


## 2.laplacian
#It calculates the Laplacian of the image
laplacian = cv2.Laplacian(img,cv2.CV_64F)

