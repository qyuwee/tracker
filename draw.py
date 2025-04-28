import cv2 as cv
import numpy as np

blank = np.zeros((500,500, 3), dtype='uint8')

cv.rectangle(blank, (150,150), (350, 350), (0, 255, 0), thickness=2)
cv.putText(blank, 'i love u', (190,250), cv.FONT_ITALIC, 1.0, (0,52,0), 2)
cv.imshow('Rectangle', blank)

cv.waitKey(0)