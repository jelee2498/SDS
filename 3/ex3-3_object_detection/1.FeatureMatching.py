import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np

-1 Feature matching

# cv2.imread(fileName, flag)
# flag option#1 :cv2.IMREAD_COLOR (1)
# flag option#2 :cv2.IMREAD_GRAYSCALE (0)
# flag option#3 :cv2.IMREAD_UNCHANGED (-1)
Image = cv2.imread('./mingky.jpg', 0)
objectImage = cv2.imread('./mingkyDoll.jpg', 0)

# calculate keypoints
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(objectImage, None)
kp2, des2 = orb.detectAndCompute(Image, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
matches = bf.match(des1, des2)

# sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:6]

img3 = cv2.drawMatches(objectImage, kp1, Image, kp2, good_matches, None, flags=2)

plt.figure(dpi=150)
plt.imshow(img3)

plt.show()