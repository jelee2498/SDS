import matplotlib.pyplot as plt
from skimage import io, filters, util
import numpy
from skimage import morphology

# load image file
image = io.imread('./lena_gray.gif')

for i in range(3):
    # define window size (3, 5, 7)
    MedianFilterWindowSize = 3 + 2 * i
    # make window of size MedianFilterWindowSzie
    MedianFilterWindow = morphology.square(MedianFilterWindowSize)

    # plot kernel
    ax = plt.subplot(1, 3, i+1)
    ax.imshow(filters.median(image, MedianFilterWindow), cmap='gray')
    plt.title('WindowSize = ' + str(MedianFilterWindowSize))

    i += 1

plt.show()