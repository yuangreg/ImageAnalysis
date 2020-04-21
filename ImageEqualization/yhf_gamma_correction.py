import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# load the original image
original = cv2.imread('./data/webgirl.png')
image_stack = []
for gamma in np.arange(0.5, 4, 0.5):
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.imwrite('./data_gamma/webgirl_{}.jpg'.format(gamma), adjusted)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    if len(image_stack)>0:
        image_stack = np.hstack([image_stack, adjusted])
    else:
        image_stack = adjusted
cv2.imwrite('./result/compare_gamma.jpg', image_stack)



