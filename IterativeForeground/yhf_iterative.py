import cv2
import numpy as np
import matplotlib.pyplot as plt
from InterativeMask import IterativeMask

img = cv2.imread("./data/cat.jpg")
getMask = IterativeMask(img)
mask_user = getMask.run()
plt.imshow(mask_user, cmap='gray')
plt.title('User input mask')
plt.show()

mask = 2*np.ones(img.shape[:2],np.uint8)
# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[mask_user == 0] = 0
mask[mask_user == 255] = 1

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
plt.imshow(mask, cmap='gray')
plt.title('Predicted Mask')
plt.show()

img = img*mask[:,:,np.newaxis]
cv2.imwrite("./yhf_forground.jpg", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Foreground Image')
plt.show()