
"""
2. Prewitt
"""

import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float
import scipy
from scipy import ndimage
import numpy as np


image = io.imread('./stripe.jpg')

# all channels aren't necessary for edge detection
image = image[:, :, 0]

# change the data type of image from uint8 to float64
image = img_as_float(image)

# edge filtering

VerticalEdges = filters.prewitt_v(image)
HorizontalEdges = filters.prewitt_h(image)

# plot

plt.figure(figsize=(15,3))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(VerticalEdges, cmap='gray')
plt.colorbar()
plt.title('Vertical Edges')

plt.subplot(1, 4, 3)
plt.imshow(HorizontalEdges, cmap='gray')
plt.colorbar()
plt.title('Horizontal Edges')

plt.subplot(1, 4, 4)
plt.imshow(VerticalEdges + HorizontalEdges, cmap='gray')
plt.colorbar()
plt.title('Vert + Horiz')

plt.show()