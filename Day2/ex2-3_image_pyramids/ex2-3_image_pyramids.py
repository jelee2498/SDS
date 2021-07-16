""" Image pyramid """
import cv2
import matplotlib.pyplot as plt



""" Load dataset"""
# Read image
img = cv2.imread('./sample.jpg', cv2.IMREAD_GRAYSCALE)



""" Gaussian pyramid """
# Gaussian pyramid
level = 4
g_pyr = []

for current_level in range(level):
    if current_level < 1:
        g = cv2.pyrDown(img)
        g_pyr.append(g)
    else:
        g = cv2.pyrDown(g_pyr[current_level-1])
        g_pyr.append(g)

plt.subplot(151), plt.imshow(img, 'gray')
plt.subplot(152), plt.imshow(g_pyr[0], 'gray')
plt.subplot(153), plt.imshow(g_pyr[1], 'gray')
plt.subplot(154), plt.imshow(g_pyr[2], 'gray')
plt.subplot(155), plt.imshow(g_pyr[3], 'gray')
plt.show()



""" Laplacian pyramid """
# Laplacian pyramid
level = 4
l_pyr = []

for current_level in range(level):
    if current_level < 1:
        fg = cv2.pyrUp(g_pyr[current_level])
        lap = cv2.subtract(img, fg)
        l_pyr.append(lap)
    else:
        fg = cv2.pyrUp(g_pyr[current_level])
        lap = cv2.subtract(g_pyr[current_level-1], fg)
        l_pyr.append(lap)

plt.subplot(151), plt.imshow(img, 'gray')
plt.subplot(152), plt.imshow(l_pyr[0], 'gray')
plt.subplot(153), plt.imshow(l_pyr[1], 'gray')
plt.subplot(154), plt.imshow(l_pyr[2], 'gray')
plt.subplot(155), plt.imshow(l_pyr[3], 'gray')
plt.show()



""" Reconstruction using pyramid """
# Reconstruction using pyramids
level = 4

r_pyr = g_pyr[level-1]
for current_level in range(level):
    r_pyr = cv2.pyrUp(r_pyr)
    r_pyr = cv2.add(r_pyr, l_pyr[level-current_level-1])

plt.subplot(121), plt.imshow(img, 'gray')
plt.subplot(122), plt.imshow(r_pyr, 'gray')
plt.show()
