# Image Blending through Gaussian Pyramids

Note about the operation with uint8.

1. cv2.pyrUp and cv2.pyrDown requires input to be "uint8"
2. Gaussian Laplacian is the difference between original image and Gaussian filtered image after downsampling and upsampling.
The substraction should be done with "signed" integers