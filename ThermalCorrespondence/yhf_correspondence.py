import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from findTiePoints import GetTiePoints


# Read RGB and Thermal Images
img_rgb_ori = cv2.imread("./images/DJI_0020.jpg")
img_rgb = img_rgb_ori.copy()
img_thermal_ori = cv2.imread("./images/DJI_0019_R.JPG")
img_thermal = img_thermal_ori.copy()

# Interactive tool to find tie points
thermal_obj = GetTiePoints(img_thermal)
points_thermal = thermal_obj.run()

cv2.namedWindow('Reference', cv2.WINDOW_NORMAL)
h, w, c = img_thermal.shape
cv2.resizeWindow('Reference', int(480 / h * w), 480)
cv2.imshow('Reference', img_thermal)
rbg_obj = GetTiePoints(img_rgb)
points_rgb = rbg_obj.run()

distance_rgb = cv2.norm(points_rgb[0], points_rgb[1], cv2.NORM_L2)
distance_thermal = cv2.norm(points_thermal[0] - points_thermal[1], cv2.NORM_L2)

scale = distance_thermal/distance_rgb

h, w, c = img_rgb.shape
img_rgb_resize = cv2.resize(img_rgb_ori, (int(w*scale), int(h*scale)))
x_origin = int(points_rgb[0][1]*scale) - points_thermal[0][1]
y_origin = int(points_rgb[0][0]*scale) - points_thermal[0][0]
h, w, c = img_thermal.shape
img_rgb_resize[x_origin:x_origin+h, y_origin:y_origin+w, :] = img_rgb_resize[x_origin:x_origin+h, y_origin:y_origin+w, :]*0.5 + img_thermal_ori*0.5

dict = {'scale': scale, 'x0': x_origin, 'y0': y_origin}
file = open('params', 'wb')
pickle.dump(dict, file)
file.close()


h, w, c = img_rgb.shape
cv2.namedWindow('Final Merged', cv2.WINDOW_NORMAL)
h, w, c = img_rgb_resize.shape
cv2.resizeWindow('Final Merged', int(480 / h * w), 480)
cv2.imshow('Final Merged', img_rgb_resize)
cv2.waitKey(0)
