
"""
0. Image histogram
"""

import matplotlib.pyplot as plt
from skimage import io
import numpy as np


image = io.imread('./coins.png')

# image histogram

histY, binEdges = np.histogram(image, bins=32)
# calculate relative frequency (normalized count)
histY = histY / histY.sum()
# define histX as a center between binEdges
histX = (binEdges[1:33] + binEdges[0:32]) / 2

# plot

plt.figure(figsize=(10, 5), dpi = 150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
# bar plot without width
plt.stem(histX, histY, use_line_collection=True)
plt.xlabel('Bins')
plt.ylabel('Normalized Count')
plt.title('Image Histogram')
plt.show()


