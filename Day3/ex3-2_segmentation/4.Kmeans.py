
"""
K-means clustering
"""

import matplotlib.pyplot as plt
from skimage import io
import numpy as np


try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/wallpaper.jpg').astype('int')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./wallpaper.jpg').astype('int')

sizeI, sizeJ, sizeK = image.shape

# image -> RGB vector conversion
imageRGBvect = image.reshape(-1, 3)

# Kmeans initialization
k = 3
ClusterCenters = np.random.randint(low=0, high=255, size=(k,3))

# Kmeans clustering

convergeLimit = 10
deltaMeans = 99999
maxIter = 10000
iterN = 0

while deltaMeans > convergeLimit and iterN < maxIter:
    iterN += 1

    # calculate every distance from center to points
    dist2Centers = np.zeros((sizeI*sizeJ, k))
    for i in range(k):
        temp = imageRGBvect - ClusterCenters[i, ]
        temp = np.square(temp)
        temp = np.sqrt(temp.sum(axis=1))
        dist2Centers[:, i] = temp

    # cluster assignment
    clustersAssigned = dist2Centers.argmin(axis=1)

    # cluster update
    NewClusterCenters = np.zeros((k,3))
    for i in range(k):
        clusterList = np.where(clustersAssigned == i)
        valueInList = imageRGBvect[clusterList]
        NewClusterCenters[i] = valueInList.mean(axis=0)

    # calculate convergence
    deltaMeans = NewClusterCenters - ClusterCenters
    deltaMeans = np.square(deltaMeans)
    deltaMeans = np.sqrt(deltaMeans.sum(axis=1)).sum()

    ClusterCenters = NewClusterCenters

# convert image value to ClusterCenters
ClusterCenters = np.round(ClusterCenters).astype('uint8')

imageClustered = np.zeros_like(image).astype('uint8')
imageRGBvectIdx = 0
for i in range(sizeI):
    for j in range(sizeJ):
        imageClustered[i,j] = ClusterCenters[clustersAssigned[imageRGBvectIdx]]
        imageRGBvectIdx += 1

# plot
plt.figure(figsize=(20, 10), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.imshow(imageClustered, cmap='gray', vmin=0, vmax=255)

plt.show()

