
"""
Translation-1
"""

import matplotlib.pyplot as plt
from skimage import io, transform
from scipy import ndimage
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day1/ex1-2_transformation/cameraman.tif')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./cameraman.tif')

imageTranslated = np.zeros_like(image)

# make transform matrix: x->x+15, y->y+30

# x - vertical axis
# y - horizontal axis
Tx = 15
Ty = 30

T = [[1, 0, Tx],
     [0, 1, Ty],
     [0, 0, 1]]
T_inv = np.linalg.inv(T)


# apply transform

iMax, jMax = np.shape(image)
iTranslated = 0

while iTranslated < iMax:
    jTranslated = 0
    while jTranslated < jMax:
        # find native coordinates using T_inv
        nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iTranslated,
                                                         jTranslated, 1]))
        # map value of native coordinates to the corresponding imageTranslated coordinates
        # order: the order of the spline interpolation
        imageTranslated[iTranslated, jTranslated] = ndimage.map_coordinates(image,
                                                    [[nativeI], [nativeJ]], order=1)
        jTranslated += 1
    iTranslated += 1

# check result

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTranslated, cmap='gray')
plt.show()
