# Find Correspondence Between RGB and Thermal Images

Given one RGB image and one thermal image taken from DJI XT2

1. User specify two points on thermal and the corresponding two points on RGB (Left click to select point, Right click to clear all and restart)
2. Compute the scaling factor to scale up/down the RGB image
3. Embed the thermal image into the RGB image


Parameters Saving and Loading. The mode has to be in binary 'b', i.e., 'wb' and 'rb'
```
# Saving Parameters
dict = {'scale': 1, 'x0': 2, 'y0': 3}
file = open('params', 'wb')
pickle.dump(dict, file)
file.close()

# Loading Parameters
file = open('params', 'rb')
dict = pickle.load(file)
```

![RGB](./images/sample.jpg)