
"""
Gaussian mixture model (GMM)
"""

import matplotlib.pyplot as plt
from skimage import io, color, filters
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats


try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/flower.jpg')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./flower.jpg')

# convert RGB to HSV
imageHSV = color.rgb2hsv(image)
sizeI, sizeJ, sizeK = imageHSV.shape

# image -> HSV vector conversion
# Hue(색조), Saturation(채도), Value(명도)
# most intuitive color model
imageHSVvect = imageHSV.reshape(-1, 3)

# value of Hue indicates color index (value itself is meaningless)
HueChannel = imageHSVvect[:, 0].reshape(-1, 1)

# GMM of HueChannel

clusterN = 3
GMMmodel = GaussianMixture(clusterN).fit(HueChannel)
clustersAssigned = GMMmodel.predict(HueChannel)
ClusterCenters = GMMmodel.means_
# for visualization purpose
clusteredHue = ClusterCenters[clustersAssigned]

# convert imgae value to ClusterCenters
imageHSVvectClustered = imageHSVvect.copy().transpose()

imageHSVvectClustered[0] = clusteredHue.squeeze()
# set saturation and value to be constant
imageHSVvectClustered[1] = 1
imageHSVvectClustered[2] = 1
imageHSVvectClustered = imageHSVvectClustered.transpose()

imageClusteredHSV = np.zeros_like(image).astype('float32')

# HSV vector -> image conversion
imageHSVvectIdx = 0
for i in range(sizeI):
    for j in range(sizeJ):
        imageClusteredHSV[i,j] = imageHSVvectClustered[imageHSVvectIdx]
        imageHSVvectIdx += 1

imageClustered = color.hsv2rgb(imageClusteredHSV)

# plot
plt.figure(figsize=(20, 10), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image, vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.imshow(imageClustered, vmin=0, vmax=255)

plt.show()

# plot GMM

xvec = np.arange(0, 1, .001).transpose()

plt.figure(figsize=(20, 10), dpi=150)

G1 = stats.norm(loc=ClusterCenters[0], scale=np.sqrt(GMMmodel.covariances_[0]))
distributionG1 = GMMmodel.weights_[0]*G1.pdf(xvec).transpose()

G2 = stats.norm(loc=ClusterCenters[1], scale=np.sqrt(GMMmodel.covariances_[1]))
distributionG2 = GMMmodel.weights_[1]*G2.pdf(xvec).transpose()

G3 = stats.norm(loc=ClusterCenters[2], scale=np.sqrt(GMMmodel.covariances_[2]))
distributionG3 = GMMmodel.weights_[2]*G3.pdf(xvec).transpose()

plt.hist(HueChannel, bins=128, density=True, alpha=0.3, label='Histogram')
plt.plot(xvec, distributionG1 + distributionG2 + distributionG3, alpha=0.2, linewidth=5,
         label='G1+G2+G3', color='magenta')
G1Plot = plt.plot(xvec, distributionG1, label='G1', color='b', linewidth=2)
G2Plot = plt.plot(xvec, distributionG2, label='G2', linewidth=2)
G3Plot = plt.plot(xvec, distributionG3, label='G3', linewidth=2)

plt.grid(True, which='major', axis='both')
plt.legend(fontsize=20)

plt.show()
