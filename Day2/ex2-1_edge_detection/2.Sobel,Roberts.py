
"""
Sobel and Roberts
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

# All channels aren't necessary for edge detection
image = image[:, :, 0]

# edge filtering
#---------- YOUR CODE HERE ----------#
VerticalEdges_sobel = None
HorizontalEdges_sobel = None
Edges_roberts = None
#------------------------------------#

# plot

plt.figure(figsize=(10,10))

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
plt.imshow(Edges_roberts, cmap='gray')
plt.title('Roberts Edges')

plt.show()