import ImgProc
import synthFuncs
import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution=(320,240)
time.sleep(0.1)


while True:
    keyPress = cv2.waitKey(1)
    rawCapture = np.empty((320*240*3,), dtype='uint8')
    camera.capture(rawCapture, format='bgr')
    rawCapture = rawCapture.reshape((240,320,3))
    cv2.imshow("Test", rawCapture)
    if keyPress == 32:
        masks =ImgProc.get_masks(rawCapture, 1)
        data = ImgProc.get_data(rawCapture, masks)
        synthFuncs.play_progression(synthFuncs.get_progression(data[0][1],
            data[0][3], data[0][2], data[0][4]))
    if keyPress == 27:
        break

cv2.destroyAllWindows()
