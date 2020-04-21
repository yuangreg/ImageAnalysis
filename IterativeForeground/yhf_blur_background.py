import cv2
import numpy as np
import matplotlib.pyplot as plt
from InterativeMask import InteractiveMask

# Initial User Mask
img = cv2.imread("./data/1493462248.jpg")
getMask = InteractiveMask(img)
mask_user = getMask.run()
# plt.imshow(mask_user, cmap='gray')
# plt.title('User input mask')
# plt.show()

mask = 3*np.ones(img.shape[:2],np.uint8)
# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
# 2: Probable background, 3: probable foreground
mask[mask_user == 0] = 0
mask[mask_user == 255] = 1

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Loop until obtain good foreground image
img_filterd = cv2.bilateralFilter(img,51,50,100)
while(1):
    mask, bgdModel, fgdModel = cv2.grabCut(img_filterd, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask_new = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # mask on image
    img_new = (img_filterd*0.5+mask_new[:,:,np.newaxis]*127).astype('uint8')

    getMask = InteractiveMask(img_new)
    mask_user = getMask.run()
    if(mask_user.max() == 128 and mask_user.min() == 128):
        break
    mask[mask_user == 0] = 0
    mask[mask_user == 255] = 1

img_foreground = img*mask_new[:,:,np.newaxis]
img_background = cv2.GaussianBlur(img,(51,51),0)*(1-mask_new[:,:,np.newaxis])
img_new = img_foreground + img_background
cv2.imwrite("./yhf_blur_background.jpg", img_new)