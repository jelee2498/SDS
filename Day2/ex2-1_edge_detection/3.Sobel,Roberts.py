
"""
3. Sobel and Roberts - soltuion
"""

import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float
import scipy
from scipy import ndimage
import numpy as np


image = io.imread('./dog.jpg')

# all channels aren't necessary for edge detection
image = image[:, :, 0]

# change the data type of image from uint8 to float64
image = img_as_float(image)

# edge filtering
#---------- YOUR CODE HERE ----------#
VerticalEdges_sobel = ?
HorizontalEdges_sobel = ?
PosDiagEdges_roberts = ?
NegDiagEdges_roberts = ?
#------------------------------------#

# plot

plt.figure(figsize=(10,8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(VerticalEdges_sobel, cmap='gray')
plt.title('Sobel Vertical Edges')

plt.subplot(2, 3, 3)
plt.imshow(HorizontalEdges_sobel, cmap='gray')
plt.title('Sobel Horizontal Edges')

plt.subplot(2, 3, 5)
plt.imshow(PosDiagEdges_roberts, cmap='gray')
plt.title('Positive Diagonal Edges')

plt.subplot(2, 3, 6)
plt.imshow(NegDiagEdges_roberts, cmap='gray')
plt.title('Negative Diagonal Edges')

plt.show()