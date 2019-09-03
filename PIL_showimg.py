import PIL
import numpy as np
from PIL import Image
img = PIL.Image.open('./ImageEqualization/data/webgirl.png')
img.load()
img_array = np.array(img)

# Display image.
new_image = PIL.Image.fromarray(img_array)
new_image.show()