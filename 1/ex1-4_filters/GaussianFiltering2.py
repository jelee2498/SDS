import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage import io, transform, filters, util
import scipy
from scipy import ndimage
import numpy
import math
import scipy.stats as st
from scipy import signal


# load image file
image = io.imread('./lena_gray.gif')

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


sigma = 3

for i in range(3):
    # kernel setting
    kernelSize = 5 + i * 4
    kernelRadius = (kernelSize - 1) / 2

    # for surface plot
    x = np.linspace(-kernelRadius, kernelRadius, kernelSize)
    y = np.linspace(-kernelRadius, kernelRadius, kernelSize)

    X, Y = np.meshgrid(x, y)

    # caculate Gaussian kernel
    GaussianKernel = gkern(kernelSize, sigma)

    # plot kernel
    ax = plt.subplot(2, 3, i+1, projection='3d')
    ax.plot_surface(X, Y, GaussianKernel, cmap='viridis', edgecolor='none')
    plt.title('KernelSize = ' + str(kernelSize))

    # plot filtered image
    ax = plt.subplot(2, 3, i+4)
    ax.imshow(imageConvolution(image, GaussianKernel), cmap='gray')

    i += 1

plt.show()