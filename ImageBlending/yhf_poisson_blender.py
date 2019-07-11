import cv2
import numpy as np
from GetPolygon import GetPolygon

# Read images
src = cv2.imread("data2/small_bird.png")
dst = cv2.imread("data2/farm.png")
getPoly = GetPolygon(src)

# Create a rough mask by drawing
src_mask = np.zeros(src.shape, src.dtype)
polyPoints = np.array(getPoly.run())
cv2.fillPoly(src_mask, [polyPoints], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (1200, 800)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# Save result
cv2.imwrite("./yhf_blend2.jpg", output);