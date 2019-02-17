import ImgProc
import synthFuncs
import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
rawCapture = PiRGBArray(camera)
time.sleep(0.1)

while True:
    camera.capture(rawCapture, format="bgr")
    img=rawCapture.array
    cv2.imshow("Test", img)
    keyPress = cv2.waitKey(1)
    rawCapture.truncate(0)
    if keyPress == 27:
        break

cv2.destroyAllWindows()
