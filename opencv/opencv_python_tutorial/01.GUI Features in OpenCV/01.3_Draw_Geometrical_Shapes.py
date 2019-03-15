## Packages
import numpy as np
import cv2

## Create a black image
img = np.zeros((512,512,3), np.uint8)

## Draw a line
cv2.line(img,(10,10),(490,10),(255,0,0),5) 

## Draw a rectangle
cv2.rectangle(img,(20,20),(480,80),(0,255,0),3)

## Draw a circle
cv2.circle(img,(150,150), 50, (0,0,255), -1) 	# Filled
cv2.circle(img,(350,150), 50, (0,0,255),  3)    # Outline

## Draw an ellispe
cv2.ellipse(img,(250,200),(200,100),0,0,180,(100,100,0),3)

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
