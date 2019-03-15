# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png', 0)

### Canny Edge Detection ###
#Working
#A. Noise Reduction
#Edge detection is susceptible to noise in the image
GaussianBlur = cv2.GaussianBlur(img,(5,5),0)

#B.Finding Intensity Gradient of the Image
#Sobel kernel in both horizontal and vertical direction to get first derivative in
#horizontal direction (𝐺𝑥) and vertical direction (𝐺𝑦).
#𝐸𝑑𝑔𝑒_𝐺𝑟𝑎𝑑𝑖𝑒𝑛𝑡 (𝐺) = sqrt(Gx*Gx + Gy*Gy)

#C. Non-maximum Suppression (remove any unwanted pixels)
#At every pixel, pixel is checked if it is a local maximum in its neighborhood.

#D.Hysteresis Thresholding (decides which are all edges are really edges)
#two threshold values, minVal and maxVal.
#D.1 Any edges with intensity gradient more than maxVal are sure to be edges
#D.2 those below minVal are sure to be non-edges, so discarded
#D.3 Edges or Non Edges
#Edges: they are connected to “sure-edge” pixels
#Otherwise, they are also discarded.

#cv2.canny(input_img,minVal,maxVal,aperture_size,L2gradient)
#aperture size: size of Sobel kernel used for find image gradients
#L2gradient: True(accurate) use above expression of (G)
#L2gradient: Fals, G = |Gx|+|Gy|
canny_edge = cv2.Canny(GaussianBlur, 100, 200)
