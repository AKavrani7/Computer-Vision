# -*- coding: cp1252 -*-
## Packages
import cv2  #opencv-python
import numpy as np 

## Load an image
img = cv2.imread('the-incredible-hulk.jpg')
## Warning: Even if the image path is wrong, it won’t throw any error,
## but print(img) will give you None

## Image Parameters
print(type(img)) #<type 'numpy.ndarray'>
print(img.shape) #(720, 1280, 3)
print(img.dtype) #uint8

## Display an image
#cv2.imshow(window_name,image_name) to display an image in a window.
cv2.imshow('image',img)
## Gray_Scale
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## Save an image
#cv2.imwrite(file_name,image_name) to save an image
cv2.imwrite('Gray-the-incredible-hulk.jpg',gray_image)

k = cv2.waitKey(0)
cv2.destroyAllWindows()
