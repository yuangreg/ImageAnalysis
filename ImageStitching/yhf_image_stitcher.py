import glob, cv2
from ImageStitcher import ImageStitcher

directory = './carmel/'
file_format = '*.png'
total = directory+file_format
files = glob.glob(total)
files.sort()

imageStitcher = ImageStitcher()
ind = 0
coord = None
for file in files:
    print(file)
    img = cv2.imread(file)
    if ind==0:
        img_old = img
    elif ind == 1:
        imageStitcher.initialize(img_old, img)
        img_merge, mask1_merge, mask2_merge = imageStitcher.merge()
        if img_merge is not None:
            img_old = img_merge
        else:
            break
    else:
        imageStitcher.initialize(img_old, img, mask1 = mask2_merge)
        img_merge, mask1_merge, mask2_merge = imageStitcher.merge()
        if img_merge is not None:
            img_old = img_merge
        else:
            break
    ind += 1

cv2.imwrite("./yhf_final.png", img_old)

