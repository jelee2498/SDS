import matplotlib.pyplot as plt
from skimage import io, transform
import scipy
from scipy import ndimage
import numpy
import math


fpath = ''
image = io.imread(fpath + 'cameraman.tif')
imageTranslated = numpy.zeros(numpy.shape(image))

# make transform
iMax, jMax = numpy.shape(image)

Tx = ?
Ty = ?
Translation = \
    [[1, 0, Tx],
     [0, 1, Ty],
     [0, 0, 1]]


Theta = math.radians(?)
Rotation = \
    [[math.cos(Theta), -math.sin(Theta), 0],
     [math.sin(Theta), math.cos(Theta), 0],
     [0, 0, 1]]


Shx = ?
Shy = ?
Shear = [[1, Shx, 0],
     [Shy, 1, 0],
     [0, 0, 1]]


T = numpy.dot(?, numpy.dot(?, ?))
T = numpy.linalg.inv(T)

# apply transform
iTransformed = 0
while iTransformed < iMax:

    jTransformed = 0
    while jTransformed < jMax:

        nativeI, nativeJ, temp = numpy.dot(T, numpy.transpose([iTransformed, jTransformed, 1]))
        imageTranslated[iTransformed, jTransformed] = ndimage.map_coordinates(image, [[nativeI], [nativeJ]], order=1)

        jTransformed = jTransformed + 1
    iTransformed = iTransformed + 1

# Check result
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTranslated, cmap='gray')
plt.show()
