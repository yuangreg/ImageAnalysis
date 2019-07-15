import cv2
import numpy as np
from BoundaryDetection import BoundaryDetection

img = cv2.imread("./data/Fine-Art.jpg")
bd = BoundaryDetection(img)
points = bd.run()

src = np.float32(points)
h, w, c = img.shape
margin = 10


# Boundary order: follows from BoundaryDetection
TL = [margin, margin]
TR = [w-1-margin, margin]
BR = [w-1-margin, h-1-margin]
BL = [margin, h-1-margin]
temp = [TL, BL, BR, TR]
dst = np.float32(temp)

H, _ = cv2.findHomography(src, dst, method=0)
temp = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H)


planar_image = cv2.warpPerspective(img, H, (w, h))
cv2.imwrite("./yhf_planar_image.png", planar_image)

