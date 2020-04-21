import numpy as np
import cv2

# Original Image
img_bgr = cv2.imread('./data/webgirl.png')
# Initialize the CLAHE adaptive equalizer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Method CLAHE
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_clahe = clahe.apply(img_gray)
cv2.imwrite('./result/image_clahe.jpg', img_clahe)

# Method CLAHE LAB
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)
l_eq = clahe.apply(l)
img_lab_eq = cv2.merge((l_eq, a, b))
image_rgb = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
cv2.imwrite('./result/image_clahe_lab.jpg', image_rgb)


# Method Equalization YUV
img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imwrite('./result/image_clahe_yuv.jpg', img_output)