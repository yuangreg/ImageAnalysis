import cv2
import pickle, glob

# Read parameters
file = open('params', 'rb')
dict = pickle.load(file)
scale = dict['scale']
x0 = dict['x0']
y0 = dict['y0']

# Create image list
img_rbg_list = []
for file in glob.glob("./100MEDIA/RGB/*.*"):
    img_rbg_list.append(file)
img_rbg_list.sort()

img_thermal_list = []
for file in glob.glob("./100MEDIA/thermal/*.*"):
    img_thermal_list.append(file)
img_thermal_list.sort()

for i in range(len(img_rbg_list)):
    print(i)
    img_rgb = cv2.imread(img_rbg_list[i])
    img_thermal = cv2.imread(img_thermal_list[i])

    h, w, c = img_rgb.shape
    img_rgb_resize = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))

    h, w, c = img_thermal.shape
    img_rgb_resize[x0:x0+h, y0:y0+w, :] = img_thermal

    outfile = img_rbg_list[i].replace("RGB", "combine")
    cv2.imwrite(outfile, img_rgb_resize)
