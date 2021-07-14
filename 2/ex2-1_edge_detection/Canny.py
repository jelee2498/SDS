import matplotlib.pyplot as plt
from skimage import io, filters, feature


# load image file
image = io.imread('./sample2.jpg')
image = image[:, :, 1]

# plot original image

plt.figure(figsize=(15, 15))

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

