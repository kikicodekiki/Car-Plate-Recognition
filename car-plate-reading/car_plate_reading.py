import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr

img = cv2.imread('/Users/ekaterinastoyanova/PycharmProjects/Car-Plate-Recognition/images/picture1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# adding less sound to the photo so as to read the symbols better
img_filter = cv2.bilateralFilter(gray, 11, 17, 17)
# fix the contours
edges = cv2.Canny(img_filter, 100, 200)
contour = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
contour = imutils.grab_contours(contour)
contour = sorted(contour, key=cv2.contourArea, reverse=True)[:8] # sort the contours in reverse to put them on top
# getting only the last 8 contours

# finding the contours that contain the car plate
pos = None
for c in contour:
    approx = cv2.approxPolyDP(c, 5, True)

plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
plt.show()