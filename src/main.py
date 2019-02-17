import ImgProc
import synthFuncs
import cv2
import time
import numpy as np
from picamera import PiCamera
import wiringpi

camera = PiCamera()
camera.resolution=(320,240)
time.sleep(0.1)

num_effective_contours = 5

while True:
    keyPress = cv2.waitKey(1)
    rawCapture = np.empty((320*240*3,), dtype='uint8')
    camera.capture(rawCapture, format='bgr')
    rawCapture = rawCapture.reshape((240,320,3))
    cv2.imshow("Test", rawCapture)
    if keyPress == 32:
        masks =ImgProc.get_masks(rawCapture, 1)
        top_indices = [(-1,-1)] * num_effective_contours
        x, y, num_masks = np.shape(masks) 
        for i in range(num_masks - 1):
            current = num_effective_contours-1
            print("%d, %d, %d, %d" % (current, len(top_indices), i, num_masks))
            while(current >= 0 and ImgProc.get_size(masks[:,:,i]) > top_indices[current][0]):
                tmp = top_indices[current]
                top_indices[current] = (i, ImgProc.get_size(masks[:,:,i]))
                top_indices.insert(current + 1, tmp)
                top_indices = top_indices[0:num_effective_contours]
                current = current - 1
        newMasks = np.zeros((x,y,num_effective_contours), dtype='uint8')
        for i in range( len(top_indices) ):
            if i < num_masks and top_indices[i][0] != -1:
                newMasks[:,:,i] = masks[:,:,i]
            else:
                newMasks = newMasks[:,:,0:i-1]
        data = ImgProc.get_data(rawCapture, newMasks)
        for d in data:
            synthFuncs.play_progression(synthFuncs.get_progression(d[1],
                d[3], d[2], d[4]))
            # time.sleep(.08)
    if keyPress == 27:
        break

cv2.destroyAllWindows()
