# -*- coding: cp1252 -*-
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

## Background Subtractor(extrat moving foreground)##
#From static camera
#Image = moving foreground + static background

## Three Algorithm
#1.BackgroundSubtractorMOG and 2.BackgroundSubtractorMOG2
#Gaussian Mixture-based Background/Foreground Segmentation Algorithm
#Paper: “An improved adaptive background mixture model for real-time tracking with shadow detection” by P. KadewTraKuPong and R. Bowden in 2001.
#K Gaussian distributions (K = 3 to 5), The weights of the mixture represent the time proportions that those colours stay in the scene

#1.a cv2.createBackgroundSubtractorMOG(): create a background Object
fgbg = cv2.createBackgroundSubtractorKNN()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
#cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()







llWindows()
