# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png')

### Transformation Function
#F1. cv2.warpAffine, 2x3 transformation matrix
#F2. cv2.warpPerspective, 3x3 transformation matrix

## 1. Scaling: Resizing of the image.
#Preferable interpolation methods are
#1.a cv2.INTER_AREA for shrinking
#1.b cv2.INTER_CUBIC (slow)
#1.c cv2.INTER_LINEAR for zooming.
#1.d (Default) cv2.INTER_LINEAR for all resizing
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#OR
height, width = img.shape[:2]
scaling = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

### 2. Translation
#Translation is the shifting of object’s location. let it be (𝑡𝑥, 𝑡𝑦), you can create
#𝑀 =
#[︂ 1 0 𝑡𝑥 ]
#[ 0 1 𝑡𝑦 ]︂
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
translation = cv2.warpAffine(img,M,(cols,rows)) ##(cols,rows) is the shape of output image

### 3.Rotation
#Rotation of an image for an angle 𝜃 is achieved by the transformation matrix of the form
#𝑀 =
#[︂ 𝑐𝑜𝑠𝜃 −𝑠𝑖𝑛𝜃 ]
#[ 𝑠𝑖𝑛𝜃  𝑐𝑜𝑠𝜃 ]

rows,cols = img.shape
#rotates the image by 90 degree with respect to center
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
rotation = cv2.warpAffine(img,M,(cols,rows))

### 4.Affine Transformation
#All parallel lines in the original image will still be parallel in the output image
#cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.
#we need three points from input image and their corresponding locations in output image
rows,cols = img.shape
input_pts = np.float32([[50,50],[200,50],[50,200]])
output_pts = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(input_pts,output_pts)
Aff_Transform = cv2.warpAffine(img,M,(cols,rows))


### 5.Perspective Transform
#Straight lines will remain straight even after the transformation
#cv2.getPerspectiveTransform will create a 3x3 matrix which is to be passed to cv2.warpAffine.
#4 points on the input image and corresponding points on the output image

rows,cols,ch = img.shape
input_pts = np.float32([[56,65],[368,52],[28,387],[389,390]])
output_pts = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(input_pts,output_pts)
perspective_transform = cv2.warpPerspective(img,M,(300,300))
