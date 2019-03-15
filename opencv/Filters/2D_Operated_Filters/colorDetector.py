import cv2
import numpy as np

def detectBlue(input_image):
    #BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Range of Blue Color
    blue_lower = np.array([110,50,50],np.uint8)
    blue_upper = np.array([130,255,255],np.uint8)
    #Finding the Range of Blue Color in the Image
    blue = cv2.inRange(hsv,blue_lower,blue_upper)

    #Morphological Transformation,dilation
    kernal = np.ones((5,5),"uint8")
    blue = cv2.dilate(blue,kernal)
    res = cv2.bitwise_and(img,img,mask = blue)
    #Tracking the Blue color
    (image,contours,hierarchy)= cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return res,contours

def detectRed(input_image):
    #BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Range Of Red Color
    red_lower = np.array([0,50,50],np.uint8)
    red_upper = np.array([10,255,255],np.uint8)
    #finding the range of red color in the image
    red = cv2.inRange(hsv,red_lower,red_upper)
    
    #Morphological Transformation,dilation
    kernal = np.ones((5,5),"uint8")
    red = cv2.dilate(red,kernal)
    res = cv2.bitwise_and(img,img,mask = red)
    #Tracking the Red color
    (image,contours,hierarchy)= cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return res,contours

def detectOrange(input_image):
    #BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Range of Orange Color
    orange_lower = np.array([0,208,186],np.uint8)
    orange_upper = np.array([47,255,255],np.uint8)
    #finding the range of orange color in the image
    orange = cv2.inRange(hsv,orange_lower,orange_upper)
    
    #Morphological Transformation,dilation
    kernal = np.ones((5,5),"uint8")
    orange = cv2.dilate(orange,kernal)
    res = cv2.bitwise_and(img,img,mask = orange)

    #Tracking the Orange color
    (image,contours,hierarchy)= cv2.findContours(orange,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return res,contours

def detectYellow(input_image):
    #BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Range of Yellow Color
    yellow_lower = np.array([20,100,100],np.uint8)
    yellow_upper = np.array([30,255,255],np.uint8)
    #finding the range of yellow color in the image
    yellow = cv2.inRange(hsv,yellow_lower,yellow_upper)
    
    #Morphological Transformation,dilation
    kernal = np.ones((5,5),"uint8")
    yellow = cv2.dilate(yellow,kernal)
    res = cv2.bitwise_and(img,img,mask = yellow)

    #Tracking the Yellow color
    (image,contours,hierarchy)= cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return res,contours

def detectGreen(input_image):
    #BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #defining the range of Green color
    green_lower = np.array([22,60,150],np.uint8)
    green_upper = np.array([110,255,255],np.uint8)
    #finding the range of green color in the image
    green = cv2.inRange(hsv,green_lower,green_upper)

    #Morphological Transformation,dilation
    kernal = np.ones((5,5),"uint8")
    green = cv2.dilate(green,kernal)
    res = cv2.bitwise_and(img,img,mask = green)

    #Tracking the Green color
    (image,contours,hierarchy)= cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return res,contours
