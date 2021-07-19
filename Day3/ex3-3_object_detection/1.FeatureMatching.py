
"""
Feature matching
"""

import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np


# cv2.imread(fileName, flag)
# flag option#1 :cv2.IMREAD_COLOR (1)
# flag option#2 :cv2.IMREAD_GRAYSCALE (0)
# flag option#3 :cv2.IMREAD_UNCHANGED (-1)

Image = cv2.imread('./SDS/Day3/ex3-3_object_detection/mingky.jpg', 0)
objectImage = cv2.imread('./SDS/Day3/ex3-3_object_detection/mingkydoll.jpg', 0)
#---------- YOUR CODE HERE ----------#
"""
Hint: use cv2.imread()
"""
# objectImage = None
#------------------------------------#
if isinstance(Image, np.ndarray):  # running in Colab
    pass
else:  # running in Pycharm
    Image = cv2.imread('./mingky.jpg', 0)
    objectImage = cv2.imread('./mingkydoll.jpg', 0)
    #---------- YOUR CODE HERE ----------#
    """
    Hint: use cv2.imread()
    """
    # objectImage = None
    #------------------------------------#

# calculate keypoints

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(objectImage, None)
kp2, des2 = orb.detectAndCompute(Image, None)

# create BFMatcher object

bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
matches = bf.match(des1, des2)

# sort them in the order of their distance

matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:6]

img3 = cv2.drawMatches(objectImage, kp1, Image, kp2, good_matches, None, flags=2)

plt.figure(dpi=150)
plt.imshow(img3)

plt.show()

