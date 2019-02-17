import cv2
import numpy as np


def get_colors(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hist = cv2.calcHist([hsv_img], channels=[0], mask=mask, histSize=[12], ranges=[0, 179])
    return hist


def get_intensity(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hist = cv2.calcHist([hsv_img], channels=[2], mask=mask, histSize=[20], ranges=[0, 255])
    return hist


def get_saturation(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hist = cv2.calcHist([hsv_img], channels=[1], mask=mask, histSize=[2], ranges=[0, 255])
    return hist


def get_size(img, mask):
    return np.sum(mask) 


def get_noisiness(img, mask):
    # Pretty much straight copied from https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i_height, i_width = gray_img.shape()
    kernel = [[-.125, -.125, -.125],
              [-.125, 1, -.125],
              [-.125, -.125, -.125]]
    pad = 1
    convolve_img = cv2.copyMakeBorder(gray_img, pad, pad, pad, pad, cv2.BORDER_WRAP)
    output = np.zeros((i_height, i_width), dtype="float32")
    for y in np.arange(pad, i_height + pad):
        for x in np.arange(pad, i_width + pad):
            roi = convolve_img[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * kernel).sum()

            output[y-pad, x-pad] = k




def get_img_maps(img, num_blocks):
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = np.shape(greyscale)
    img_height = int(img_height)
    img_width = int(img_width)
    depth = num_blocks ** 2
    retval = np.zeros((img_height, img_width, depth), dtype="uint8")
    height_step = img_height // num_blocks
    width_step = img_width // num_blocks
    for i in range(num_blocks):
        for j in range(num_blocks):
            this_depth = (i * (num_blocks) + j)
            retval[i * height_step:((i + 1) * height_step), j * width_step: ((j+1) * width_step), this_depth] = \
            np.ones((height_step, width_step), dtype="uint8")
    return retval



    return retval

img = cv2.imread("../index.jpeg")
masks = get_img_maps(img, 16)
x, y, z = np.shape(masks)
print (get_noisiness(np.random.rand(20,20), None))
# for i in range(z):
#     cv2.imshow("img", np.multiply(img, np.expand_dims(masks[::, ::, i], axis=2)))
#     tmp_hist = get_colors(img, masks[::,::,i])
#     print (i)
#     cv2.waitKey(100)



