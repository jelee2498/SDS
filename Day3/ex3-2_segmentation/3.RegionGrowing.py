
"""
Region growing
"""

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import seaborn as sns


try:  # running in Colab
    image = io.imread('./SDS/Day3/ex3-2_segmentation/medtest.png').astype('int')
except FileNotFoundError:  # running in Pycharm
    image = io.imread('./medtest.png').astype('int')

# region growing

seedI = 187
seedJ = 345

maxDiff = 50

sizeI, sizeJ = image.shape

segMap = np.zeros_like(image)
segMap[seedI, seedJ] = 1

# breadth-first search(BFS): explores all node at the present depth
BFSQueue = np.zeros((1000, 2)).astype('int')
BFSQueue[0, ] = [seedI, seedJ]

SearchDirections = np.array([[0, -1],
                             [1, 0],
                             [0, 1],
                             [-1, 0]])

head = 0
tail = 1
while head != tail:  # iterate until no more tails are left
    headI = BFSQueue[head, 0]
    headJ = BFSQueue[head, 1]

    for i in range(4):  # 4 possible directions
        newI = int(BFSQueue[head, 0] + SearchDirections[i, 0])
        newJ = int(BFSQueue[head, 1] + SearchDirections[i, 1])

        # BFS search condition
        if 0 <= newI < sizeI and 0 <= newJ < sizeJ and segMap[newI, newJ] == 0:
            # segmentation condition
            if np.absolute(image[seedI, seedJ] - image[newI, newJ]) <= maxDiff:
                print('[Push] : New tail node [' + str(newI) + ', ' + str(newJ) +
                      '] will be inserted, Tail Index = ' + str(tail + 1))
                segMap[newI, newJ] = 1 # assign as foreground
                BFSQueue[tail] = newI, newJ
                tail += 1

    # pop
    print('[Pop] : Head node [' + str(headI) + ', ' + str(headJ) + '] will be deleted')
    BFSQueue[0:tail-1] = BFSQueue[1:tail]
    tail -= 1

# plot

plt.figure(figsize=(20, 10), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.plot(seedJ, seedI, 'r.')

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
my_cmap = sns.light_palette(sns.xkcd_rgb["purplish blue"], input="his",
                            as_cmap=True, reverse=False)
plt.imshow(segMap, cmap=my_cmap, alpha=0.5, vmin=0, vmax=1)
plt.colorbar()
plt.plot(seedJ, seedI, 'r.')

plt.show()
