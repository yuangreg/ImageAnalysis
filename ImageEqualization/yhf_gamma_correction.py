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
cv2.imwrite('./compare_gamma.jpg', image_stack)


######################################################################
# Use interative tool to get the foreground mask
from InterativeMask import IterativeMask
img = cv2.imread('./data_gamma/webgirl_2.0.jpg')
getMask = IterativeMask(img)
mask_user = getMask.run()

mask = 3*np.ones(img.shape[:2],np.uint8)
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


#########################################################################
# Blending the images from mask
from yhf_pyramid_blender import PyramidBlender

src = cv2.imread("data_gamma/webgirl_2.5.jpg")
dst = cv2.imread("data_gamma/webgirl_0.5.jpg")

mask_3c = np.repeat(mask_new[:, :, np.newaxis], 3, axis=2)
imageBlender = PyramidBlender(src, dst, mask = 1-mask_3c, layers=6)
output_image = imageBlender.run()
cv2.imwrite("./yhf_blend.jpg", output_image)



