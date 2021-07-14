import matplotlib.pyplot as plt
from skimage import io, filters, util, morphology, measure
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


PIXEL_MAX = 255
# Load image file
fpath = ''
image = io.imread(fpath + 'lena_gray.gif')


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

# Original Image
ax = plt.subplot(3, 4, 1)
ax.imshow(image, cmap='gray')
plt.title('OriginalImage')
plt.xlabel('NRMSE:' + str(round(measure.compare_nrmse(image, image, norm_type='mean'), 4))
           + ', SSIM:'+str(round(measure.compare_ssim(image, image), 4)))

# Gaussian Noise Reduction
ax = plt.subplot(3, 4, 2)
ax.imshow(imageGaussianNoise, cmap='gray')
plt.title('GaussianNoise')
plt.xlabel('NRMSE:' + str(round(measure.compare_nrmse(image, imageGaussianNoise, norm_type='mean'), 4))
           + ', SSIM:'+str(round(measure.compare_ssim(image, imageGaussianNoise), 4)))

ax = plt.subplot(3, 4, 3)
filteredImage = imageConvolution(imageGaussianNoise, GaussianKernel)
ax.imshow(filteredImage, cmap='gray')
plt.title('GaussianFiltering')
plt.xlabel('NRMSE:' + str(round(measure.compare_nrmse(image, filteredImage, norm_type='mean'), 4))
           + ', SSIM:'+str(round(measure.compare_ssim(image, filteredImage), 4)))

ax = plt.subplot(3, 4, 4)
filteredImage = filters.median(imageGaussianNoise, MedianFilterWindow)
ax.imshow(filteredImage, cmap='gray')
plt.title('MedianFiltering')
plt.xlabel('NRMSE:' + str(round(measure.compare_nrmse(image, filteredImage, norm_type='mean'), 4))
           + ', SSIM:'+str(round(measure.compare_ssim(image, filteredImage), 4)))

# Salt & Pepper Noise Reduction
''' Write Your Answer here '''

# Speckle Noise Reduction
''' Write Your Answer here '''

plt.show()
