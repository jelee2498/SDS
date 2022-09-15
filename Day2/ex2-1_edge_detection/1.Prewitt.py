
"""
1. Prewitt - solution
"""

import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float
import scipy
from scipy import ndimage
import numpy as np

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

            #------------------------------------#
            j += 1
        i += 1

    return ConvImage


image = io.imread('./stripe.jpg')

# all channels aren't necessary for edge detection
image = image[:, :, 0]

# change the data type of image from uint8 to float64
image = img_as_float(image)

# define Prewitt kernel

# horizontal edge kernel
prewitt_h = np.array([[ 1/3,  1/3,  1/3],
                      [   0,    0,    0],
                      [-1/3, -1/3, -1/3]])
# vertical edge kernel
prewitt_v = np.array([[ 1/3,    0,  -1/3],
                      [ 1/3,    0,  -1/3],
                      [ 1/3,    0,  -1/3]])

# make edge images

edge_h = imageConvolution(image, prewitt_h)
edge_v = imageConvolution(image, prewitt_v)

# plot

plt.figure(figsize=(12,3))

# plot 3 images (original image, horizontal edge, vertical edge)
#---------- YOUR CODE HERE ----------#
"""
Hint: Use plt.subplot
"""

#------------------------------------#

plt.show()
