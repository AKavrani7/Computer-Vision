import sys
import os
import cv2

def load_images(folder):
    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))    
        if img is not None:
            images.append(img)
    
    return images