import cv2
import numpy as np

img = cv2.imread('Naruto_Uzumaki.png', cv2.IMREAD_COLOR)

## 1.Access pixel values and modify them

#Pixel Operation
px = img[100,100]
print(px)

px = img[100:103,100:102]
# 100:103 - height 
# 100:102 - width
# output in form of height X width
# Somewhat printing the matrix is inverse of matrix's rules
print(px)

#accessing RED value
print(img.item(10,10,2))
#modifying RED value
img.itemset((10,10,2),255)
print(img.item(10,10,2))


## 2.Access image properties

#It returns a tuple of number of rows, columns and channels
print(img.shape)
#Total number of pixels
print(img.size)
#Image datatype
print(img.dtype)

## 3. Image ROI

#ROI is obtained using Numpy indexing.
roi_img = img[100:200,100:200]

## 4.Splitting and Merging Image Channels

# Now the concept if you split the image in b,g,r values or in three channels
# here b is the image represent blue channel or in image where blue's intensity
# is less the darker will be the b image.
b,g,r = cv2.split(img) #Costly operation
cv2.imshow('image_Split_B',b)
img = cv2.merge((b,g,r))

b = img[:,:,0] #Numpy idexing is efficient

cv2.waitKey(0)
cv2.destroyAllWindows()
