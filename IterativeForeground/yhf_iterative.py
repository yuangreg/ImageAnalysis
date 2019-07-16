import cv2
import numpy as np
import matplotlib.pyplot as plt
from InterativeMask import IterativeMask

# Initial User Mask
img = cv2.imread("./data/cat.jpg")
getMask = IterativeMask(img)
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
while(1):
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask_new = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # mask on image
    img_new = (img*0.5+mask_new[:,:,np.newaxis]*127).astype('uint8')

    getMask = IterativeMask(img_new)
    mask_user = getMask.run()
    if(mask_user.max() == 128 and mask_user.min() == 128):
        break
    mask[mask_user == 0] = 0
    mask[mask_user == 255] = 1

img_foreground = img*mask_new[:,:,np.newaxis]
cv2.imwrite("./yhf_forground.jpg", img_foreground)