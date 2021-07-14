import matplotlib.pyplot as plt
from skimage import io, transform
from scipy import ndimage
import numpy


Tx = 15
Ty = 30

# make transform matrix: x->x+15, y->y+30

tform_inv = transform.AffineTransform(translation=[-Tx, -Ty])
print('')
print('Transformation matrix')
print(tform_inv.params)

# apply transform

# skimage.transform.warp(image, inverse_map)
# inverse_map: inverse coordinate map, which transforms coordinates in the ouput images into
# their corresponding coordinates in the input image
imageTranslated = transform.warp(image, tform_inv)

# check result

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTranslated, cmap='gray')
plt.show()
