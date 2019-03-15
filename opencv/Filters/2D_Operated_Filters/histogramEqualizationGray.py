import numpy as np
import cv2

def histogram_equalization_Gray(img):

    #Step1 Convert the image's colorSpace from BGR to GrayScale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original",grayImg)
    
    #Step2 Calculate cdf
    hist_gImg = cv2.calcHist([grayImg],[0],None,[256],[0,256])
    cdf_gImg = np.cumsum(hist_gImg)

    #Step3 Equalize the distribution
    gray = np.around(np.subtract(cdf_gImg, np.amin(cdf_gImg)))
    cv2.divide(gray, grayImg.size, gray)
    cv2.multiply(gray, 255, gray)

    #Step4 Final Touch
    newGrayImg = gray[grayImg.ravel()].reshape(grayImg.shape)

    return newGrayImg
