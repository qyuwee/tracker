import cv2 as cv
import numpy as np
img = cv.imread('C:/dev/materials/opencv/photos/cats.jpg')

# Averaging
average = cv.blur(img, (3,3))

# Gaussian blur
gauss = cv.GaussianBlur(img, (3,3), 0)

# Median blur
median = cv.medianBlur(img, 3)

# Bilateral
bilateral = cv.bilateralFilter(img, 5, 15, 15)


cv.imshow('median', median)
cv.imshow('defImg', average)
cv.imshow('gauss blur', gauss)




cv.waitKey(0)