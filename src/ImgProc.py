import cv2
import numpy as np

# TODO pass highest bucket back out
def get_colors(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hsv_img = (hsv_img[:,:,0] + 7) % 180
    hist = cv2.calcHist([hsv_img], channels=[0], mask=mask, histSize=[12], ranges=[0, 179])
    return np.argmax(hist)

# TODO Get the max value here
def get_intensity(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hist = cv2.calcHist([hsv_img], channels=[2], mask=mask, histSize=[20], ranges=[0, 255])
    return np.argmax(hist)


def get_saturation(img, mask):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = np.multiply(hsv_img, np.expand_dims(masks[::, ::, i], axis=2))
    hist = cv2.calcHist([hsv_img], channels=[1], mask=mask, histSize=[2], ranges=[0, 255])
    return np.argmax(hist)


def get_size(mask):
    return np.sum(mask) 


def get_noisiness(img, mask):
    i, j = np.where(mask)
    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                          np.arange(min(j), max(j) + 1),
                          indexing='ij')
    sub_img = img[indices]

    # Pretty much straight copied from https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    gray_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    i_height, i_width = np.shape(gray_img)
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

            output[y-pad, x-pad] = abs(k)
    return int(np.average(output) * 6/43.0)


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


def get_masks(img, target_depth):
    """Finds all contours in an image and returns a list of masks, all of which are filled in rectangular
    bounding boxes of each contour.

    target_depth is the lowest depth of the contour to evaluate
    """
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(im, 100, 200)
    contours, hiearchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    print(hiearchy)

    # Filter out all contours that are too deep in the hiearchy
    valid_contours = []
    for i in range(len(contours)):
        depth = 0
        parent = hiearchy[0][i][3]
        while parent != -1:
            depth += 1
            parent = hiearchy[0][parent][3]
        if depth <= target_depth:
            valid_contours.append(contours[i])

    masks = np.zeros(im.shape + (len(valid_contours),), dtype="uint8")

    # Generates rectanglular 
    for i in range(len(valid_contours)):

        # Finds the dims of the bounding rect
        x, y, w, h = cv2.boundingRect(valid_contours[i])

        # Draws the rectangle on the mask
        tmp = np.zeros(im.shape, dtype='uint8')
        cv2.rectangle(tmp, (x, y), (x + w, y + h), color = 1, thickness = cv2.FILLED)
        masks[:,:,i] = tmp
    return masks


def get_data(img, masks):
    i_height, i_width, num_contours = np.shape(masks)
    data = []
    for i in range(num_contours):
        mask = masks[:, :, i]
        data.append((get_size(mask),
                     get_colors(img, mask),
                     get_intensity(img, mask),
                     get_saturation(img, mask),
                     get_noisiness(img, mask)))
    return data

img = cv2.imread("Boshi!.jpg")
#img = np.random.randint(0,255,size=(500, 500, 3), dtype="uint8")
#img[:, :, 2] = np.random.randint(255, size=(500,500), dtype="uint8")
masks = get_masks(img, target_depth = 2)
x, y, z = np.shape(masks)
# print (get_noisiness(np.random.randint(255, size=(1000, 1000, 3), dtype="uint8"), None))
# print (get_noisiness(np.ones((1000, 1000, 3), dtype="uint8"), None))
for i in range(z):
    cv2.imshow("img", np.multiply(img, np.expand_dims(masks[::, ::, i], axis=2)))
    print("Image %d" % (i + 1))
    tmp_hist = get_colors(img, masks[::, ::, i])
    print("\tColor Range: %d" % tmp_hist)
    tmp_intensity = get_intensity(img, masks[::,::,i])
    print("\tIntensity: %d" % tmp_intensity)
    tmp_saturation = get_saturation(img, masks[::,::,i])
    print("\tSaturation: %d" % tmp_saturation)
    tmp_size = get_size(masks[::,::,i])
    print("\tSize: %d" % tmp_size)
    print("\tNoisiness: %d" % (get_noisiness(img, masks[::, ::, i])))
    if cv2.waitKey(1) == 27:
        break
