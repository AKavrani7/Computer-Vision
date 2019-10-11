import numpy as np               
import cv2
from matplotlib import pyplot as plt

from src.working_files import load_images
from src.preprocess import image_enhancement, isolataion, watershed_algorithm

image_source = 'input_images'
save_output = 'output_images'

######### 1. Load/Read Images ########
print("Step 1 Load Images")
input_images = load_images(image_source)
print("total no: " + str(len(input_images)))

######### 2. Pre Processing ########
print("Step 2. Pre-Processing")
for num, img in enumerate(input_images):
  print("image no: " + str(num))
  cv2.imshow('input_image', img)

  # image enhancement
  print("\tImage enhancement")
  gray, otsu_thresh, closing = image_enhancement(img)
  #cv2.imshow('gray', gray)
  #cv2.imshow('otsu_thresh', otsu_thresh)
  #cv2.imshow('closing', closing)

  # isolataion
  print("\tIsolation")
  isolated = isolataion(closing)
  sure_bg, sure_fg, unknown = isolated 
  #cv2.imshow('sure_bg', sure_bg)
  #cv2.imshow('sure_fg', sure_fg)
  #cv2.imshow('unknown', unknown)

  # watershed
  print("\tWatershed")
  output_image, marker = watershed_algorithm(isolated, img)
  
  cv2.imshow('output_image', output_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  ######### 3. Save the output Image ##########
  print("saving the output image")
  cv2.imwrite(str(save_output)+ '/output_image_' + str(num+1) + ".jpg", output_image)

print("Coin detection Complete")






