import matplotlib.pyplot as plt
from skimage import io, transform
import scipy
from scipy import ndimage
import numpy
import math


imageTransformed = np.zeros_like(image)

# make transformation matrix

iMax, jMax = np.shape(image)

Tx = 2 * (iMax - 1)
Ty = 0
Translation = [[1, 0, Tx],
               [0, 1, Ty],
               [0, 0, 1]]

Theta = math.radians(90)
Rotation = [[math.cos(Theta), -math.sin(Theta), 0],
            [math.sin(Theta), math.cos(Theta), 0],
            [0, 0, 1]]

Shx = 0
Shy = 1
Shear = [[1, Shx, 0],
         [Shy, 1, 0],
         [0, 0, 1]]

T = reduce(np.dot, [Translation, Rotation, Shear])
T_inv = np.linalg.inv(T)

# apply transform

iTransformed = 0

while iTransformed < iMax:
    jTransformed = 0
    while jTransformed < jMax:
        # find native coordinates using T_inv
        nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iTransformed,
                                                         jTransformed, 1]))
        # map value of native coordinates to the corresponding imageTranslated coordinates
        # order: the order of the spline interpolation
        imageTransformed[iTransformed, jTransformed] = ndimage.map_coordinates(image,
                                                    [[nativeI], [nativeJ]], order=1)
        jTransformed += 1
    iTransformed += 1

# check result

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTransformed, cmap='gray')
plt.show()