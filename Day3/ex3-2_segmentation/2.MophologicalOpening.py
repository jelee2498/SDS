
"""
Morphological opening
- structural element(SE): kernel or mask
- erosion: erode foreground pixels -> remove noise (e.g., small blobs)
- dilation: dilate foreground pixels
- opening: erosion -> dilation
"""

import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/rice.png')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./rice.png')

# background estimation using mophological opening
background = morphology.opening(image, morphology.disk(15))

# segmentation

image2 = image - background
OstuTh = filters.threshold_otsu(image2, nbins=256)
imageSegmented = image2 >= OstuTh

# plot

plt.figure(figsize=(10, 20), dpi=150)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.subplot(2, 2, 2)
plt.imshow(background, cmap='gray', vmin=0, vmax=255)
plt.title('Background')
plt.subplot(2, 2, 3)
plt.imshow(image2, cmap='gray', vmin=0, vmax=255)
plt.title('Original - Background')
plt.subplot(2, 2, 4)
plt.imshow(imageSegmented, cmap='gray', vmin=0, vmax=1)
plt.title('Segmentation')

plt.show()
