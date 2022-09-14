
"""
Rotation - 1
"""

import matplotlib.pyplot as plt
from skimage import io, transform
import scipy
from scipy import ndimage
import numpy as np
import math


image = io.imread('./cameraman.tif')
imageRotated = np.zeros_like(image)

# make transformation matrix: 30 degree

Theta = math.radians(30)
T = [[math.cos(Theta), -math.sin(Theta), 0],
     [math.sin(Theta), math.cos(Theta), 0],
     [0, 0, 1]]
T_inv = np.linalg.inv(T)

# apply transform

iMax, jMax = np.shape(image)
iRotated = 0

while iRotated < iMax:
    jRotated = 0
    while jRotated < jMax:
        # find native coordinates using T_inv
        nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iRotated, jRotated, 1]))
        # map value of native coordinates to the corresponding imageTranslated coordinates
        # order: the order of the spline interpolation
        imageRotated[iRotated, jRotated] = ndimage.map_coordinates(image, [[nativeI], [nativeJ]], order=1)
        jRotated += 1
    iRotated += 1

# check result

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageRotated, cmap='gray')
plt.show()
