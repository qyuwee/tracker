import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('C:/dev/materials/opencv/photos/cat.jpg')

# Blank
blank = np.zeros((400, 400), dtype='uint8')

# Rectangle
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

# Circle
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# Converting to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)

# Edge Cascade (границы)
canny = cv.Canny(img, 125, 175)

# Corners
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)
corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
if corners is not None:
    for x, y in numpy.float32(corners).reshape(-1, 2):
        cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

# Dilating (использовать с canny желательно) (увеличение границ)
dilated = cv.dilate(canny, (3,3), iterations=1)

# Eroding (dilated to default) (уменьшение границ)
eroded = cv.erode(dilated, (3,3), iterations=1)

# Resize (use interpolation if def->smaller: area     else: linear or cubic(the slowest but best quality)
resized = cv.resize(img, (500,500))

# Cropping
cropped = img[50:200, 200:400]

# Translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x --> Right
# y -- > Down

translated = translate(img, 100, 100)

# Rotation


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(img, 45)

# Flipping 0 -> vertical, 1 -> horizontal, -1 - both axes
flip = cv.flip(img, 0)

# Contouring TREE->hierarchies, EXTERNAL->EXTERNAL, LIST->ALL_COUNTERS. CHAIN_APPROX_NONE->ALL_POINTS,
# CHAIN_APPROX_SIMPLE->2_POINTS
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s), found')

# Another canny (binarezation) (but canny better)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

# Drowing contours
blank = np.zeros((500,500, 3), dtype='uint8')
cv.drawContours(blank, contours, -1, (0,0,255), 2)

# Spliting
b,g,r = cv.split(img)

# Merging
merged = cv.merge([b, g, r])

# Averaging
average = cv.blur(img, (3,3))

# Gaussian blur
gauss = cv.GaussianBlur(img, (3,3), 0)

# Median blur
median = cv.medianBlur(img, 3)

# Bilateral
bilateral = cv.bilateralFilter(img, 5, 15, 15)

# Bitwise AND
bitwise_and = cv.bitwise_and(rectangle, circle)

# Bitwise OR
bitwise_or = cv.bitwise_or(rectangle, circle)

# Bitwise XOR
bitwise_xor = cv.bitwise_xor(rectangle, circle)

# Bitwise NOT
bitwise_not = cv.bitwise_not(rectangle)

# Masking
mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img,img,mask=mask)

# Gray histogram
gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# Colour histogram
mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img,mask=mask)
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')

colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color = col)
    plt.xlim([0,256])

plt.show()

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Haar cascades
haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)


cv.imshow('img', flip)

cv.waitKey(0)
