
"""
1. Thresholding
"""

import matplotlib.pyplot as plt
from skimage import io
import numpy as np


image = io.imread('./coins.png')

# thresholding
#---------- YOUR CODE HERE ----------#
"""
Hint: Segmenet coins from background using threshold "100"
"""
image_thresholded = ?
#------------------------------------#

plt.figure(figsize=(10, 5), dpi = 150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(image_thresholded, cmap='gray')
plt.show()