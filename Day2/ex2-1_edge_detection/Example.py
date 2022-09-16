
"""
Example (Lung nodule segmentation) - solution
"""

import nibabel as nib
import numpy as np # linear algebra
import pandas as pd # reading and processing of tables
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage.util import montage as montage2d
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc


img = nib.load("lung_clippled.nii").get_fdata()
gt = nib.load("gt_clippled.nii").get_fdata()

plt.imshow(img.T, cmap='gray')
plt.imshow(gt.T, alpha=0.2, cmap='jet')
plt.show()

def segmentLung(image, gt):
    """
    This funtion segments the lungs from the given 2D slice.
    """
    im = image.copy() # don't change the input

    f, plots = plt.subplots(2, 3, figsize=(13, 7))
    plots = plots.flatten()

    plots[0].axis('off')
    plots[0].imshow(im, cmap=plt.cm.bone)
    plots[0].imshow(gt, cmap='spring', alpha=0.2)
    plots[0].set_title('Original CT Image\n+GT')

    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -600
    plots[1].axis('off')
    plots[1].imshow(binary, cmap=plt.cm.bone)
    plots[1].set_title('First Threshold')

    cleared = clear_border(binary)

    '''
    Step 2: Label the image.
    '''
    label_image = label(cleared)

    '''
    Step 3: Take largest label as lung
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    plots[2].axis('off')
    plots[2].imshow(binary, cmap=plt.cm.bone)
    plots[2].set_title('Largest label')

    '''
    Step 4: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    plots[3].axis('off')
    plots[3].imshow(binary, cmap=plt.cm.bone)
    plots[3].set_title('Fill holes')

    '''
    Step 5: Superimpose the binary mask on the input image.
    '''
    ZERO_VALUE = -2048
    #---------- YOUR CODE HERE ----------#
    """
    Hint: Use '==' logical operator
    """

    #------------------------------------#
    plots[4].axis('off')
    plots[4].imshow(im, cmap=plt.cm.bone)
    plots[4].set_title('Segmented Lungs Image')

    '''
    Step 6: Canny edge detection
    '''
    #---------- YOUR CODE HERE ----------#
    """
    Hint: Fine-tune the parameters
    """
    sigma = ?
    low_T = ?
    high_T = ?
    cannyImage = feature.canny(im, sigma, low_threshold=low_T,
                               high_threshold=high_T)
    #------------------------------------#
    plots[5].axis('off')
    plots[5].imshow(cannyImage, cmap=plt.cm.bone)
    plots[5].set_title("Edge Detection Result")

    plt.show()

segmentLung(img.T, gt.T)
