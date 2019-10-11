import numpy as np
import cv2

def image_enhancement(img):

    '''
    Image contains two classes of pixel foreground pixel and background pixel.
    1. Gray: Converts the RGB scale to gray scale
    
    2. Otsu Binarization: It calculates the optimum threshold value.
        Example: Consider a bimodal image(the histogram of an image has two peaks).
                 For that image, we can take an approximate value in 
                 the middle of those peaks as threshold value.
    
    3. Morphological Transformation: Due to thresholding, binary images are distored by noise and texture.
        Morphological image processing is a collection of non-linear operations related to the morphology
        (shape, convexity, connectivity and geodesic distance) of features in an image.

        Note: Black(0) and white(1)
        Image: foreground_pixel(white) and background_pixel(black)
        https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html

        A. Erosion: A pixel in the original image (either 1 or 0) will be considered 1 only if all the 
                    pixels under the kernel is 1, otherwise it is eroded (made to zero).
        
        B. Dilation: Opposite of erosion.

        C. Opening: erosion followed by dilation.

        D. Closing: Dilation followed by Erosion

    '''

    # Gray (single channel image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    shape = gray.shape
    print("\tshape of the image after gray scale conversion: " + str(shape))

    # Otsu Binarization
    ret, otsu_thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Morphological Transform
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    opening = cv2.morphologyEx(otsu_thresh,cv2.MORPH_OPEN,kernel,iterations=1)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 13)
    
    return gray, otsu_thresh, closing

def isolataion(img):

    '''
    Ensure the foreground and background portions of the image
    # 1. Sure Background(dilation): find the area which we are sure they are not coins.

    # 2. Sure Foreground: Extract the area which we are sure they are coins.
        Distance Transform: The result of the transform is a graylevel image
            except that the graylevel intensities of points inside foreground regions 
            are changed to show the distance to the closest boundary from each point.
        
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # sure background area
    sure_bg = cv2.dilate(img,kernel,iterations=3)

    # sure foreground area
    dist_transform = cv2.distanceTransform(img,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.986*dist_transform.max(),255,0)

    # unknown area
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    return sure_bg, sure_fg, unknown

def watershed_algorithm(isolated, img):

    '''
    The watershed algorithm is a classic algorithm used for segmentation and 
    is especially useful when extracting touching or overlapping objects in images.

    1. find contours:
    Three params (source image, contour retrieval mode, contour approximation method)
    contours: Contours is a Python list of all the contours in the image. Each individual contour 
        is a Numpy array of (x,y) coordinates of boundary points of the object.
    
    2. Create Markers
    Foreground: centre(white), rest(black) 
    Background: centre(black), rest(white)
    Marker = Foreground + Background
    Marker: in_coin(white), across_coin(black), rest(gray)
    '''

    sure_bg, sure_fg, unknown = isolated 
    shape = sure_fg.shape

    # contours
    contours, hierarchy = cv2.findContours(sure_fg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # numpy array for markers. convert image to 32 bit using dtype paramter
    marker = np.zeros((shape[0], shape[1]),dtype = np.int32)
    marker = np.int32(sure_fg) + np.int32(sure_bg)

    # Note: cv2 version doesnt offer cv2.markers function so here labelling of markers done by contours

    print("\tlen of contours: " + str(len(contours)))
    for id in range(len(contours)):
        cv2.drawContours(marker,contours,id,id+2, -1)
    
    marker = marker + 1
    # set value of unknown area equal to zero or black
    marker[unknown==255] = 0

    # watershed
    cv2.watershed(img, marker)
    img[marker==-1]=(0,0,255)

    return img, marker