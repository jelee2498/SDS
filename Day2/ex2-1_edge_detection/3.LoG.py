
"""
Laplacian of Gaussian
"""

import matplotlib.pyplot as plt
from skimage import io, filters
import scipy
from scipy import ndimage
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day2/ex2-1_edge_detection/sample2.jpg')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./sample2.jpg')

image = image[:, :, 0].astype('float64')

plt.figure(figsize=(18, 18))

plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for Loopi in range(3):
    sigma = 1 + 2 * Loopi
    # compute LoG image
    LoGImage = ndimage.gaussian_laplace(image, sigma=sigma)

    # edge image
    EdgeImage = np.zeros_like(image)
    # threshold for edge detection
    th = np.absolute(LoGImage).mean() * 0.75
    iMax, jMax = image.shape

    i = 1
    while i < iMax - 1:
        j = 1
        while j < jMax - 1:
            # extract (3,3) patch
            patch = LoGImage[i - 1:i + 2, j - 1:j + 2]
            if LoGImage[i, j] > 0: # for positive LoG
                # if LoGImage[i, j] is edge, minimum LoG in pathces should have different sign with it
                zeroCross = LoGImage[i, j] * patch.min() < 0
            else:  # for negative LoG
                # if LoGImage[i, j] is edge, maximum LoG in pathces should have different sign with it
                zeroCross = LoGImage[i, j] * patch.max() < 0

            # two condition for LoGImage[i, j] to be edge
            # 1. Difference btw maximum and minumum LoG in patch should be over threshold
            # 2. LoGImage[i, j] is zero-crossing point
            if patch.max() - patch.min() > th and zeroCross:
                EdgeImage[i, j] = 1

            j += 1
        i += 1


    # plot LoG and after-touched image

    plt.subplot(3, 3, 3*Loopi+2)
    plt.imshow(LoGImage, cmap='gray')
    plt.title('LoG Image (sigma=' + str(sigma) + ')')

    plt.subplot(3, 3, 3*Loopi+3)
    plt.imshow(EdgeImage, cmap='gray')
    plt.title('Edge Image')

    Loopi += 1

plt.show()

