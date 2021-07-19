
"""
Thresholding
"""

import matplotlib.pyplot as plt
from skimage import io
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/coins.png')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./coins.png')

# thresholding
#---------- YOUR CODE HERE ----------#
"""
Hint: Segmenet coins from background using threshold "100"
Return: thresholded Image 
"""
image_thresholded = None
#------------------------------------#

plt.figure(figsize=(10, 5), dpi = 150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_thresholded, cmap='gray')
plt.show()