
"""
Example (noise filtering)
"""

import matplotlib.pyplot as plt
from skimage import io, filters, util, morphology, metrics
import numpy
from scipy import signal
import math

def gkern(kernlen, nsig):
    # Return 2D Gaussian Kernel
    gkern1d = signal.gaussian(kernlen, std=nsig).reshape(kernlen, 1)
    kernel = numpy.outer(gkern1d, gkern1d)

    return kernel/kernel.sum()

def imageConvolution(image,kernel):

    ConvImage = numpy.zeros(numpy.shape(image))
    KernelSizeI, KernelSizeJ = numpy.shape(kernel)
    KernelRadius = int((KernelSizeI - 1)/2)

    iMax, jMax = numpy.shape(image)
    i = KernelRadius
    while i < iMax - KernelRadius:
        j = KernelRadius
        while j < jMax - KernelRadius:
            # convolution operation
            ConvImage[i, j] = numpy.multiply(image[i-KernelRadius:i+KernelRadius+1, j-KernelRadius:j+KernelRadius+1], kernel).sum()

            j = j + 1

        i = i + 1

    return ConvImage.astype('uint8')


image = io.imread('./lena_gray.gif')

PIXEL_MAX = 255

# make noise image

imageGaussianNoise = util.noise.random_noise(image, mode='gaussian')*PIXEL_MAX
imageGaussianNoise = imageGaussianNoise.astype('uint8')

imageSaltAndPepper = util.noise.random_noise(image, mode='s&p')*PIXEL_MAX
imageSaltAndPepper = imageSaltAndPepper.astype('uint8')

imageSpeckleNoise = util.noise.random_noise(image, mode='speckle')*PIXEL_MAX
imageSpeckleNoise = imageSpeckleNoise.astype('uint8')

# Kernel Definition

kernelSize = 3
sigma = 3
GaussianKernel = gkern(kernelSize, sigma)
MedianFilterWindow = morphology.square(kernelSize)

# original image

plt.figure(figsize=(18, 15))

ax = plt.subplot(3, 4, 1)
ax.imshow(image, cmap='gray')
plt.title('Original Image')
# NRMSE: normalized root mean squared error
# if compare_nrmse is deprecated, then use normalized_root_mse instead
# SSIM: structural similarity index measure
# if compare_ssim is deprecated, then use structural_similarity instead
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, image, normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, image), 4)))

# Gaussian nosie reduction

ax = plt.subplot(3, 4, 2)
ax.imshow(imageGaussianNoise, cmap='gray')
plt.title('GaussainNoise')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageGaussianNoise, normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageGaussianNoise), 4)))

ax = plt.subplot(3, 4, 3)
filteredImage = imageConvolution(imageGaussianNoise, GaussianKernel)
ax.imshow(filteredImage, cmap='gray')
plt.title('GaussainFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, filteredImage,normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, filteredImage), 4)))

ax = plt.subplot(3, 4, 4)
filteredImage = filters.median(imageGaussianNoise, MedianFilterWindow)
ax.imshow(filteredImage, cmap='gray')
plt.title('MedianFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, filteredImage, normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, filteredImage), 4)))

# salt & pepper noise reduction

"""
Your code here
"""

# speckle noise reduction

"""
Your code here
"""

plt.show()