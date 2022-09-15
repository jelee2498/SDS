
"""
5. Canny
"""

import matplotlib.pyplot as plt
from skimage import io, filters, feature, img_as_float


image = io.imread('./dog.jpg')

# all channels aren't necessary for edge detection
image = image[:, :, 0]

# change the data type of image from uint8 to float64
image = img_as_float(image)

# plot original image

plt.figure(figsize=(15, 3))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for i, sigma in enumerate([1, 3, 5]):
    # Canny
    CannyEdge = feature.canny(image, sigma)

    plt.subplot(1, 4, i+2)
    plt.imshow(CannyEdge, cmap='gray')
    plt.title('Canny Edge (sigma= ' + str(sigma) + ')')

    sigma += 2

plt.show()

