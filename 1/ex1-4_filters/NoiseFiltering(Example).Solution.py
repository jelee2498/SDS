import matplotlib.pyplot as plt
from skimage import io, filters, util, morphology, measure
import numpy
from scipy import signal
import math

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

    return ConvImage.astype('uint8')

PIXEL_MAX = 255

# noise generation
imageGaussianNoise = util.noise.random_noise(image, mode='gaussian') * PIXEL_MAX
imageGaussianNoise = imageGaussianNoise.astype('uint8')
imageSaltAndPepper = util.noise.random_noise(image, mode='s&p') * PIXEL_MAX
imageSaltAndPepper = imageSaltAndPepper.astype('uint8')
imageSpeckleNoise = util.noise.random_noise(image, mode='speckle') * PIXEL_MAX
imageSpeckleNoise = imageSpeckleNoise.astype('uint8')

# # check results
# plt.subplot(1, 4, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original')
#
# plt.subplot(1, 4, 2)
# plt.imshow(imageGaussianNoise, cmap='gray')
# plt.title('Gaussian')
#
# plt.subplot(1, 4, 3)
# plt.imshow(imageSaltAndPepper, cmap='gray')
# plt.title('Salt and Pepper')
#
# plt.subplot(1, 4, 4)
# plt.imshow(imageSpeckleNoise, cmap='gray')
# plt.title('Speckle')
# plt.show()

# kernel definition
kernelSzie = 3
sigma = 3
GaussianKernel = gkern(kernelSzie, sigma)
MedianFilterWindow = morphology.square(kernelSzie)

# original image

plt.figure(figsize=(18, 18))

ax = plt.subplot(3, 4, 1)
ax.imshow(image, cmap='gray')
plt.title('OriginalImage')
# NRMSE: normalized root mean squared error
# if compare_nrmse is deprecated, then use normalized_root_mse instead
# SSIM: structural similarity index measure
# if compare_ssim is deprecated, then use structural_similarity instead
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, image,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, image), 4)))

# Gaussian nosie reduction

ax = plt.subplot(3, 4, 2)
ax.imshow(image, cmap='gray')
plt.title('GaussainNoise')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageGaussianNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageGaussianNoise), 4)))

ax = plt.subplot(3, 4, 3)
filteredImage = imageConvolution(imageGaussianNoise, GaussianKernel)
ax.imshow(filteredImage, cmap='gray')
plt.title('GaussainFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageGaussianNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageGaussianNoise), 4)))

ax = plt.subplot(3, 4, 4)
filteredImage = filters.median(imageGaussianNoise, MedianFilterWindow)
ax.imshow(filteredImage, cmap='gray')
plt.title('MedianFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageGaussianNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageGaussianNoise), 4)))

# salt & pepper noise reduction

ax = plt.subplot(3, 4, 6)
ax.imshow(image, cmap='gray')
plt.title('Salt & pepper noise')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSaltAndPepper,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSaltAndPepper), 4)))

ax = plt.subplot(3, 4, 7)
filteredImage = imageConvolution(imageSaltAndPepper, GaussianKernel)
ax.imshow(filteredImage, cmap='gray')
plt.title('GaussainFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSaltAndPepper,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSaltAndPepper), 4)))

ax = plt.subplot(3, 4, 8)
filteredImage = filters.median(imageSaltAndPepper, MedianFilterWindow)
ax.imshow(filteredImage, cmap='gray')
plt.title('MedianFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSaltAndPepper,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSaltAndPepper), 4)))

# speckle noise reduction

ax = plt.subplot(3, 4, 10)

ax.imshow(image, cmap='gray')
plt.title('Speckle noise')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSpeckleNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSpeckleNoise), 4)))

ax = plt.subplot(3, 4, 11)
filteredImage = imageConvolution(imageSpeckleNoise, GaussianKernel)
ax.imshow(filteredImage, cmap='gray')
plt.title('GaussainFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSpeckleNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSpeckleNoise), 4)))

ax = plt.subplot(3, 4, 12)
filteredImage = filters.median(imageSpeckleNoise, MedianFilterWindow)
ax.imshow(filteredImage, cmap='gray')
plt.title('MedianFiltering')
plt.xlabel('NRSME: ' + str(round(metrics.normalized_root_mse(image, imageSpeckleNoise,
                                                             normalization='mean'), 4)) +
           ', SSIM: ' + str(round(metrics.structural_similarity(image, imageSpeckleNoise), 4)))

plt.show()
