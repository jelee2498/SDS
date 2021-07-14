
import matplotlib.pyplot as plt
from skimage import io, filters, feature
import scipy
from scipy import ndimage
import numpy as np


#-1 Prewitt

# # load image file
# image = io.imread('./sample2.jpg')
# image = image[:, :, 1]

# # edge filtering
#
# VerticalEdges = filters.prewitt_v(image)
# HorizontalEdges = filters.prewitt_h(image)
#
# # plot
# plt.subplot(1, 4, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(1, 4, 2)
# plt.imshow(VerticalEdges, cmap='gray')
# plt.title('Vertical Edges')
#
# plt.subplot(1, 4, 3)
# plt.imshow(HorizontalEdges, cmap='gray')
# plt.title('Horizontal Edges')
#
# plt.subplot(1, 4, 4)
# plt.imshow(VerticalEdges + HorizontalEdges, cmap='gray')
# plt.title('Vert + Horiz')
#
# plt.show()


#-2 Lapalcian of Gaussain (LoG)

# # load image file
# image = io.imread('./sample2.jpg')
# image = image[:, :, 1].astype('float64')
#
# plt.figure(figsize=(18, 18))
#
# plt.subplot(3, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
#
# for Loopi in range(3):
#     sigma = 1 + 2 * Loopi
#     # edge filtering
#     LoGImage = ndimage.gaussian_laplace(image, sigma=sigma)
#
#     # after touch
#     EdgeImage = np.zeros_like(image)
#     th = np.absolute(LoGImage).mean() * 0.75
#     iMax, jMax = image.shape
#
#     i = 1
#     while i < iMax - 1:
#         j = 1
#         while j < jMax - 1:
#             patch = LoGImage[i - 1:i + 2, j - 1:j + 2]
#             if LoGImage[i, j] > 0:
#                 zeroCross = LoGImage[i, j] * patch.min() < 0
#             else:
#                 zeroCross = LoGImage[i, j] * patch.max() < 0
#
#             if patch.max() - patch.min() > th and zeroCross:
#                 EdgeImage[i, j] = 1
#
#             j += 1
#         i += 1
#
#     # plot LoG and after-touched image
#
#     plt.subplot(3, 3, 3*Loopi+2)
#     plt.imshow(LoGImage, cmap='gray')
#     plt.title('LoG Image (sigma=' + str(sigma) + ')')
#
#     plt.subplot(3, 3, 3*Loopi+3)
#     plt.imshow(EdgeImage, cmap='gray')
#     plt.title('Edge Image')
#
#     Loopi += 1
#
# plt.show()


#-3 Canny

# load image file
image = io.imread('./sample2.jpg')
image = image[:, :, 1]

# plot original image

plt.figure(figsize=(15, 15))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for i, sigma in enumerate([1, 3, 5]):
    # Canny
    CannyEdge = feature.canny(image, sigma)

    plt.subplot(1, 4, i+2)
    plt.imshow(CannyEdge, cmap='gray')
    plt.title('Canny Edge (sigma= ' + str(sigma) + ')')

    sigma += 2

plt.show()









