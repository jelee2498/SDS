
"""
3. Morphological opening
- erosion: erode foreground pixels -> remove noise (e.g., small blobs)
- dilation: dilate foreground pixels
- closing: dilation -> erosion
- opening: erosion -> dilation
"""

import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import numpy as np


#-1 closing -> improve binary image

image = io.imread('./coins.png')

# segmentation

OstuTh = filters.threshold_otsu(image, nbins=256)
imageThresholded = image >= OstuTh

#---------- YOUR CODE HERE ----------#
"""
Tune size of disk to improve binary image (1-10)
"""
sizeDisk = ?
imageClosed = morphology.closing(imageThresholded, morphology.disk(sizeDisk))
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

#-2 opening -> segment image

image = io.imread('./rice.png')

#---------- YOUR CODE HERE ----------#
"""
Tune size of disk to estimate background (10-20)
"""
sizeDisk = ?
background = morphology.opening(image, morphology.disk(sizeDisk))
imageSegmented = image - background
#------------------------------------#

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