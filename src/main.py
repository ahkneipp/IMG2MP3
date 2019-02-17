import ImgProc
import synthFuncs
import cv2
import numpy as np

cameraCapture = cv2.VideoCapture(0)
while cameraCapture.isOpened():
    success, img = cameraCapture.read()
    cv2.imshow("Test", img)
    keyPress = cv2.waitKey(0)
    success = False
    start = True
    if keyPress == 27:
        break

    if success:
        cv2.imshow("Test", img)
        start = False
    elif not start:
        break

cv2.destroyAllWindows()
cameraCapture.release()
