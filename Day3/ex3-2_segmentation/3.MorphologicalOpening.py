
"""
Morphological opening
- erosion: erode foreground pixels -> remove noise (e.g., small blobs)
- dilation: dilate foreground pixels
- closing: dilation -> erosion
- opening: erosion -> dilation
"""

import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import numpy as np


#-1 Closing -> improve binary image

try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/coins.png')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./coins.png')

# segmentation

OstuTh = filters.threshold_otsu(image, nbins=256)
imageThresholded = image >= OstuTh

# closing

#---------- YOUR CODE HERE ----------#
"""
Tune size of disk to improve binary image
"""
sizeDisk = None
imageClosed = morphology.closing(imageThresholded, morphology.disk(3))
#------------------------------------#

# plot

plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(imageThresholded, cmap='gray', vmin=0, vmax=1)
plt.title("Otsu's thresholding")
plt.subplot(1, 3, 3)
plt.imshow(imageClosed, cmap='gray', vmin=0, vmax=1)
plt.title('Dilation')


plt.show()


#-2 Opening -> segment image

try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/rice.png')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./rice.png')

# background estimation using mophological opening
background = morphology.opening(image, morphology.disk(15))
imageSegmented = image - background

# plot

plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(background, cmap='gray', vmin=0, vmax=255)
plt.title("Background")
plt.subplot(1, 3, 3)
plt.imshow(imageSegmented, cmap='gray', vmin=0, vmax=255)
plt.title('Dilation')

plt.show()