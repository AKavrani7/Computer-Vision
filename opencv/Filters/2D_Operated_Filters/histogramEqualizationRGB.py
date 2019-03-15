import numpy as np
import cv2

def histogram_equalization(img):

    #Step1 Split the image into r,g,b channels
    blue, green, red = cv2.split(img)  

    #Step2 Calculate cdf
    hist_blue = cv2.calcHist([blue], [0], None, [256], [0, 256])  #calculating the histogram and CDF for each histogram
    cdf_blue = np.cumsum(hist_blue)

    hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
    cdf_green = np.cumsum(hist_green)

    hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])
    cdf_red = np.cumsum(hist_red)

    #Step3 Equalize the distribution
    blue1 = np.around(np.subtract(cdf_blue, np.amin(cdf_blue)))
    cv2.divide(blue1, blue.size, blue1)
    cv2.multiply(blue1, 255, blue1)
    new_blue = blue1[blue.ravel()].reshape(blue.shape)

    green1 = np.around(np.subtract(cdf_green, np.amin(cdf_green)))
    cv2.divide(green1, green.size, green1)
    cv2.multiply(green1, 255, green1)
    new_green = green1[green.ravel()].reshape(green.shape)

    red1 = np.around(np.subtract(cdf_red, np.amin(cdf_red)))
    cv2.divide(red1, red.size, red1)
    cv2.multiply(red1, 255, red1)
    new_red = red1[red.ravel()].reshape(red.shape)

    #Step4 Merging all channels
    final_img = cv2.merge([new_blue, new_green, new_red])  
    
    return final_img
