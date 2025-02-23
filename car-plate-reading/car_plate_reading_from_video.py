import numpy as np
import cv2
import imutils
import easyocr

cap = cv2.VideoCapture('/Users/ekaterinastoyanova/PycharmProjects/Car-Plate-Recognition/videos/video1.mp4')
# extract for every frame
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(gray, 9, 9, 9)
    edges = cv2.Canny(img_filter, 100, 200)
    # get the contour
    contour = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    pos = None
    RECTANGLE_SIDES = 4
    for c in contour:
        """Try to find the largest contour that has the size of a rectangle"""
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if len(approx) == RECTANGLE_SIDES:
            pos = approx
            break
    # create a bitwise mask
    mask = np.zeros(gray.shape, dtype="uint8")
    new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
    bitwise_op = cv2.bitwise_and(img_filter, img_filter, mask=mask)
    # cut the photo
    (x, y) = np.where(bitwise_op == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    crop = gray[y1:y2, x1:x2]
    # read the text from the photo
    text = easyocr.Reader(['en'])
    text = text.readtext(crop)
    res = text[0][-2]
    inal_image = cv2.putText(frame, res, (approx[0][0][0], approx[1][0][1] + 100), cv2.FONT_HERSHEY_PLAIN,
                             3, (0, 0, 255),4)
    final_image = cv2.rectangle(frame, approx[0][0], approx[2][0], (0, 255, 0), 2)
    cv2.imshow('Result', final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

