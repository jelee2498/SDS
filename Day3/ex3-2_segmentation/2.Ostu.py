
"""
2. Otsu's thresholding
- goal is to find optimal threshold
- have to calculate for all possible thresholds --> slow process
- bad results for noisy images
"""

import matplotlib.pyplot as plt
from skimage import io, filters
import numpy as np


image = io.imread('./coins.png')

# histogram

histY, _ = np.histogram(image, range=(0, 255), bins=256)
histY = histY / histY.sum()

# Otsu's thresholding

sigmaBsqaured = np.zeros((256, 1))  # inter-class variance
t = 0  # initial threshold
while t <= 255:
    w_0 = histY[0:t+1].sum()  # probability of class '0'
    w_1 = histY[t+1:256].sum()  # probability of class '1'

    if w_0 != 0 and w_1 != 0:
        # mean intensity of class '0'
        mu_0 = np.multiply(np.linspace(0, t, t+1), histY[0:t+1]).sum() / w_0
        # mean intensity of class '1'
        mu_1 = np.multiply(np.linspace(t+1, 255, 255-t), histY[t+1:256]).sum() / w_1
        #---------- YOUR CODE HERE ----------#
        """
        Hint: calculate inter-class variance using pow() function  
        """
        sigmaBsqaured[t] = ?
        #------------------------------------#

    t += 1

# find optimal threshold which maximize the inter-class variance
OstuTh = sigmaBsqaured.argmax()
imageThresholded = image >= OstuTh

# plot

plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(imageThresholded, cmap='gray')
plt.title('Ostu Threshold = ' + str(OstuTh))

plt.show()

