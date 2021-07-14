import matplotlib.pyplot as plt
from skimage import io, filters
import scipy
from scipy import ndimage


# load image file
image = io.imread('./sample2.jpg')
image = image[:, :, 1]

# edge filtering

VerticalEdges = filters.prewitt_v(image)
HorizontalEdges = filters.prewitt_h(image)

# plot
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(VerticalEdges, cmap='gray')
plt.title('Vertical Edges')

plt.subplot(1, 4, 3)
plt.imshow(HorizontalEdges, cmap='gray')
plt.title('Horizontal Edges')

plt.subplot(1, 4, 4)
plt.imshow(VerticalEdges + HorizontalEdges, cmap='gray')
plt.title('Vert + Horiz')

plt.show()