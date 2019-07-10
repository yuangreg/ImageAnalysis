import glob, cv2

directory = './carmel/'
file_format = '*.png'
total = directory+file_format
files = glob.glob(total)
files.sort()

# Prepare image list
images = []
for file in files:
    image = cv2.imread(file)
    images.append(image)

# Create stitcher using opencv
# stitcher = cv2.createStitcher()
stitcher = cv2.createStitcherScans()
(status, pano) = stitcher.stitch(images)

if status == 0:
    # write the output stitched image to disk
    cv2.imwrite("./yhf_lowe_method.png", pano)
else:
    print("[INFO] image stitching failed ({})".format(status))

