# Foreground Segmentation

### Interactive Foreground & Background Tool
User specify sure foreground and sure background of an image
```
Class InteractiveMask()
```
Left click: Foreground

Right click: Background

### Grabcut Algorithm
```
cv2.grabCut(bgdModel, fgdModel)
```
https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html

### Iterative Mask and GrabCut
Add additional sure foreground and sure background to improve GrabCut.
```
while(1):
    InteractiveMask(image)
    cv2.grabCut(bgdModel, fgdModel)
    if (no change):
        break
```

###### Background Masking
```
yhf_iterative_foreground
```
![cat](./yhf_forground.jpg)
###### Background Blur
```
yhf_blur_background
```
![Mandy](./yhf_blur_background.jpg)

### Using Superpixel to find Foreground
```
class SlicMask()
```
