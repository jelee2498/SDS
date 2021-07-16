
"""
Prewitt
"""

import matplotlib.pyplot as plt
from skimage import io, filters
import scipy
from scipy import ndimage
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day2/ex2-1_edge_detection/sample1.jpg')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./sample1.jpg')

# All channels aren't necessary for edge detection
image = image[:, :, 0]


# define Prewitt kernel

# horizontal edge kernel
prewitt_h = np.array([[ 1/3,  1/3,  1/3],
                      [   0,    0,    0],
                      [-1/3, -1/3, -1/3]])
# vertical edge kernel
prewitt_v = np.array([[ 1/3,    0,  -1/3],
                      [ 1/3,    0,  -1/3],
                      [ 1/3,    0,  -1/3]])


# define imageConvolution

def imageConvolution(Image, Kernel):
    """
    Image convolution
    :param Image: input image
    :param Kernel: edge kernel
    :return: edge image
    """
    # output image
    ConvImage = np.zeros_like(Image)
    # size of edge kernel
    KernelSizeI, KernelSizeJ = np.shape(Kernel)
    # radius of edge kernel
    KernelRadius = int((KernelSizeI - 1) / 2)

    # size of input image
    iMax, jMax = np.shape(Image)

    # start with Image[KernelRadius, KernelRadius]
    i = KernelRadius
    while i < iMax - KernelRadius:
        j = KernelRadius
        while j < jMax - KernelRadius:
            # convolution operation
            #---------- YOUR CODE HERE ----------#
            """
            Hint: Use np.multiply() 
            """
            ConvImage[i ,j] = None

            #------------------------------------#
            j += 1
        i += 1

    return ConvImage


# make edge images

edge_h = imageConvolution(image, prewitt_h)
edge_v = imageConvolution(image, prewitt_v)


# plot

plt.figure(figsize=(10,10))

# plot 3 images (original image, horizontal edge, vertical edge)
#---------- YOUR CODE HERE ----------#
"""
Hint: Use plt.subplot()
"""


#------------------------------------#

plt.show()


#-2 Using skimage module

# edge filtering

VerticalEdges = filters.prewitt_v(image)
HorizontalEdges = filters.prewitt_h(image)

# plot

plt.figure(figsize=(10,10))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(VerticalEdges, cmap='gray')
plt.title('Vertical Edges')

plt.subplot(1, 4, 3)
plt.imshow(HorizontalEdges, cmap='gray')
plt.title('Horizontal Edges')

plt.subplot(1, 4, 4)
plt.imshow(VerticalEdges + HorizontalEdges, cmap='gray')
plt.title('Vert + Horiz')

# fix VerticalEdges + HorizontalEdges plot in order that edges are visible
#---------- YOUR CODE HERE ----------#
"""
Hint: Use np.abs()
"""

#------------------------------------#

plt.show()