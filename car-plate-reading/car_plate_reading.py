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
CAR_PLATE_SIDES = 4
# finding the contours that contain the car plate
pos = None
for c in contour:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == CAR_PLATE_SIDES:
        # found a rectangle => car plate
        pos = approx
        break
# using bitwise operations and masks to locate the car plate
mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_op = cv2.bitwise_and(img, img, mask=mask)
# find out what the car plate says
# cutting out the plate from the picture
x,y = np.where(mask == 255) # retrieve the pixels
x1, y1 = np.min(x), np.min(y) # get the starting coordinates
x2, y2 = np.max(x), np.max(y)
crop = gray[x1:x2, y1:y2]
# find out what the text says
text = easyocr.Reader(['en']) # state which language
text = text.readtext(crop) # read the text displayed licence plate
# plot the result text onto the photo
res = text[0][-2] # retrieve the needed text
label = cv2.putText(img, res, (x1 - 200, y2 + 160), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
final_image = cv2.rectangle(img, (x1, x2), (y1, y2), (255, 169, 69), 3)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.show()