import cv2
import numpy as np

# 256 X 256
img1 = cv2.imread('Add_img1.jpg')
img2 = cv2.imread('Add_img2.jpg')
img3 = cv2.imread('img.png')

### Airthmetic Operation on Images

## 1.Image Addition

#1.a Opencv addition is saturated operation
#250 + 10 = 260 => 255
add_opencv = img1 + img2
cv2.imshow('add_opencv',add_opencv)

#1.b Numpy addition is modulus operation
#250 + 10 = 260 => 260/256 = 4
x = np.uint8(img1)
y = np.uint8(img2)
add_numpy = x + y
print(add_numpy)
cv2.imshow('add_numpy',add_numpy)

#1.c Image Blending
#cv2.addWeighted(img1, alpha, img2, beta, gamma)
#weighted = alpha*img1 + beta*img2 + gamma
#alpha + beta = 1
weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow('weighted',weighted)



## 2. BitWise and mask Operation

# This includes bitwise AND, OR, NOT and XOR operations.
# They will be highly useful while extracting any part of the image
# defining and working with non-rectangular ROI etc
# create special image, of the same size as the original, with white pixels
# indicating the region to save and black pixels everywhere else.
# Such an image is called a mask.
# our mask is to apply some bitwise operations to merge the mask together
# with the original image, in such a way that the areas with white pixels in the mask are shown,
# while the areas with black pixels in the mask are not shown


# Create the basic black image 
mask = np.zeros(img3.shape, dtype = "uint8")

# Draw a white, filled rectangle on the mask image
cv2.rectangle(mask, (44, 100), (72, 140), (255, 255, 255), -1)

# Apply the mask and display the result
maskedImg = cv2.bitwise_and(img3, mask)
cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
cv2.imshow("Masked Image", maskedImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
