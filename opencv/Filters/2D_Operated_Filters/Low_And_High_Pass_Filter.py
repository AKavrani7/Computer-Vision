import numpy as np
import cv2

def low_pass_filter(img):
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    #Create Mask
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1 # Mask = 1

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    out_img = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return out_img  # Low pass filter result

def high_pass_filter(img):
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    #Create Mask
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0 # Mask = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    out_img = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return out_img  # High pass filter result
