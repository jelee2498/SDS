
"""
Noise generation
"""

import matplotlib.pyplot as plt
from skimage import io, filters, util
import numpy
from skimage import morphology


image = io.imread('./lena_gray.gif')

# make noise image
imageGaussianNoise = util.noise.random_noise(image, mode='gaussian')
imageSaltAndPepper = util.noise.random_noise(image, mode='s&p')
imageSpeckleNoise = util.noise.random_noise(image, mode='speckle')

# Check result

plt.figure(figsize=(15,3))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(imageGaussianNoise, cmap='gray')
plt.title('Gaussian')

plt.subplot(1, 4, 3)
plt.imshow(imageSaltAndPepper, cmap='gray')
plt.title('Salt and Pepper')

plt.subplot(1, 4, 4)
plt.imshow(imageSpeckleNoise, cmap='gray')
plt.title('Speckle')
plt.show()