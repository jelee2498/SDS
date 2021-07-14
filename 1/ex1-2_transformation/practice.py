
import matplotlib.pyplot as plt
from skimage import io, transform
from scipy import ndimage
import numpy as np
import math
from functools import reduce

image = io.imread('./cameraman.tif')


#-1 Translation-1

# imageTranslated = np.zeros_like(image)
#
# # make transform matrix: x->x+15, y->y+30
#
# Tx = 15
# Ty = 30
#
# T = [[1, 0, Tx],
#      [0, 1, Ty],
#      [0, 0, 1]]
# T_inv = np.linalg.inv(T)
#
#
# # apply transform
#
# iMax, jMax = np.shape(image)
# iTranslated = 0
#
# while iTranslated < iMax:
#     jTranslated = 0
#     while jTranslated < jMax:
#         # find native coordinates using T_inv
#         nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iTranslated,
#                                                          jTranslated, 1]))
#         # map value of native coordinates to the corresponding imageTranslated coordinates
#         # order: the order of the spline interpolation
#         imageTranslated[iTranslated, jTranslated] = ndimage.map_coordinates(image,
#                                                     [[nativeI], [nativeJ]], order=1)
#         jTranslated += 1
#     iTranslated += 1
#
# # check result
#
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.imshow(imageTranslated, cmap='gray')
# plt.show()


#-2 Translation-2

# Tx = 15
# Ty = 30
#
# # make transform matrix: x->x+15, y->y+30
#
# tform_inv = transform.AffineTransform(translation=[-Tx, -Ty])
# print('')
# print('Transformation matrix')
# print(tform_inv.params)
#
# # apply transform
#
# # skimage.transform.warp(image, inverse_map)
# # inverse_map: inverse coordinate map, which transforms coordinates in the ouput images into
# # their corresponding coordinates in the input image
# imageTranslated = transform.warp(image, tform_inv)
#
# # check result
#
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.imshow(imageTranslated, cmap='gray')
# plt.show()


#-3 Rotation-1

# imageRotated = np.zeros_like(image)
#
# # make transformation matrix: 30 degree
#
# Theta = math.radians(30)
# T = [[math.cos(Theta), -math.sin(Theta), 0],
#      [math.sin(Theta), math.cos(Theta), 0],
#      [0, 0, 1]]
# T_inv = np.linalg.inv(T)
#
#
# # apply transform
#
# iMax, jMax = np.shape(image)
# iRotated = 0
#
# while iRotated < iMax:
#     jRotated = 0
#     while jRotated < jMax:
#         # find native coordinates using T_inv
#         nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iRotated,
#                                                          jRotated, 1]))
#         # map value of native coordinates to the corresponding imageTranslated coordinates
#         # order: the order of the spline interpolation
#         imageRotated[iRotated, jRotated] = ndimage.map_coordinates(image,
#                                                     [[nativeI], [nativeJ]], order=1)
#         jRotated += 1
#     iRotated += 1
#
# # check result
#
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.imshow(imageRotated, cmap='gray')
# plt.show()


#-4 Rotation-2

# imageTransformed = np.zeros_like(image)
#
# # make transformation matrix: translation -> rotation -> translation
#
# iMax, jMax = np.shape(image)
#
# # translate so that center of image to be origin
# Tx = - (iMax - 1) / 2
# Ty = - (jMax - 1) / 2
# Translation = [[1, 0, Tx],
#                [0, 1, Ty],
#                [0, 0, 1]]
#
# Theta = math.radians(30)
# Rotation = [[math.cos(Theta), -math.sin(Theta), 0],
#             [math.sin(Theta), math.cos(Theta), 0],
#             [0, 0, 1]]
#
# T = reduce(np.dot, [np.linalg.inv(Translation), Rotation, Translation])
# T_inv = np.linalg.inv(T)
#
# # apply transform
#
# iTransformed = 0
#
# while iTransformed < iMax:
#     jTransformed = 0
#     while jTransformed < jMax:
#         # find native coordinates using T_inv
#         nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iTransformed,
#                                                          jTransformed, 1]))
#         # map value of native coordinates to the corresponding imageTranslated coordinates
#         # order: the order of the spline interpolation
#         imageTransformed[iTransformed, jTransformed] = ndimage.map_coordinates(image,
#                                                     [[nativeI], [nativeJ]], order=1)
#         jTransformed += 1
#     iTransformed += 1
#
# # check result
#
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.imshow(imageTransformed, cmap='gray')
# plt.show()


#-5 Example

imageTransformed = np.zeros_like(image)

# make transformation matrix

iMax, jMax = np.shape(image)

Tx = 2 * (iMax - 1)
Ty = 0
Translation = [[1, 0, Tx],
               [0, 1, Ty],
               [0, 0, 1]]

Theta = math.radians(90)
Rotation = [[math.cos(Theta), -math.sin(Theta), 0],
            [math.sin(Theta), math.cos(Theta), 0],
            [0, 0, 1]]

Shx = 0
Shy = 1
Shear = [[1, Shx, 0],
         [Shy, 1, 0],
         [0, 0, 1]]

T = reduce(np.dot, [Translation, Rotation, Shear])
T_inv = np.linalg.inv(T)

# apply transform

iTransformed = 0

while iTransformed < iMax:
    jTransformed = 0
    while jTransformed < jMax:
        # find native coordinates using T_inv
        nativeI, nativeJ, _ = np.dot(T_inv, np.transpose([iTransformed,
                                                         jTransformed, 1]))
        # map value of native coordinates to the corresponding imageTranslated coordinates
        # order: the order of the spline interpolation
        imageTransformed[iTransformed, jTransformed] = ndimage.map_coordinates(image,
                                                    [[nativeI], [nativeJ]], order=1)
        jTransformed += 1
    iTransformed += 1

# check result

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTransformed, cmap='gray')
plt.show()