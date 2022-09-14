
"""
Highboost filtering
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage import io, transform, filters, util
import scipy
from scipy import ndimage
import numpy as np
import math
import scipy.stats as st
from scipy import signal
import cv2

def gkern(kernlen, nsig):
    """
    Return 2D Gaussian kernel
    :param kernlen: number of points in the output window
    :param nsig: the standard deviation
    :return: 2D Gaussian kernel
    """
    gkern1d = signal.gaussian(kernlen, std=nsig).reshape(kernlen, 1)
    kernel = np.outer(gkern1d, gkern1d)  # compute the outer product of two vectors

    return kernel / kernel.sum()

def imageConvolution(Image, Kernel):
    """
    Image convolution
    :param Image:
    :param Kernel:
    :return: Image after convolution
    """
    ConvImage = np.zeros_like(Image)
    KernelSizeI, KernelSizeJ = np.shape(Kernel)
    KernelRadius = int((KernelSizeI - 1) / 2)

    iMax, jMax = np.shape(Image)
    i = KernelRadius

    # convolvolution within Image[KernelRadius:iMax-KernelRadius, KernelRadius:jMax-KernelRdius]
    while i < iMax - KernelRadius:
        j = KernelRadius
        while j < jMax - KernelRadius:
            # convolution operation
            ConvImage[i ,j] = np.multiply(Image[i-KernelRadius:i+KernelRadius+1,
                                          j-KernelRadius:j+KernelRadius+1], Kernel).sum()
            j += 1
        i += 1

    return ConvImage


image = io.imread('./einstein.jpg')

# kernel setting
kernelSize = 11 # odd number
kernelRadius = (kernelSize - 1) / 2
sigma = 5

# caculate Gaussian kernel
GaussianKernel = gkern(kernelSize, sigma)

# blur the image
blurred_image = imageConvolution(image, GaussianKernel)

# subtract blurred image from the original image -> high-frequency components
hf_components = cv2.addWeighted(image, 1, blurred_image, -1, 0)

plt.figure(figsize=(10,3))
for i in range(3):
    k = 2 * i

    # add high-frequency components to original image
    sharpend_image = cv2.addWeighted(image, 1, hf_components, k, 0)

    ax = plt.subplot(1, 3, i+1)
    ax.imshow(sharpend_image, cmap='gray')

plt.show()


