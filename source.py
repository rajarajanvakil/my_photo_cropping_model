import cv2
from PIL import Image
import numpy as np
import imutils

src='/home/vakilsearch/Desktop/2.jpg'

i=cv2.imread(src)
i = cv2.resize(i, (480, 500))
# Resize and convert to grayscale
img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
#bilateral filter
img = cv2.bilateralFilter(img, 9, 75, 75)
# Create black and white image based on adaptive threshold

img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)

# Median filter clears small details
img = cv2.medianBlur(img, 11)

# Add black border in case that page is touching an image border
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

edges = cv2.Canny(img, 10, 10)

cnts1 = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts1)
c = max(cnts, key=cv2.contourArea)

# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


print("x1 =", extLeft[0], "y1 = ", extTop[1])
print("x2 =", extRight[0], "y2 = ", extBot[1])



print(extLeft," ext left")
print(extRight," ext right")
print(extTop,"ext top")
print(extBot," ext bot")

cv2.rectangle(i, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (0, 255, 0), 2)
roi = i[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
cv2.imwrite('/home/Documents/cropped.jpg', roi)

cv2.imshow('cropped', roi)
cv2.imshow('regular', i)
cv2.imshow("dddd",edges)

#cv2.waitKey(0)
cv2.imshow("edges",edges)
cv2.waitKey(0)
print(c.size)
