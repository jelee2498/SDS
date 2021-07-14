
"""
Pycharm basics
"""

import cv2


img = cv2.imread('./sample.jpg')

cv2.imshow('img', img)
cv2.waitKey(0)  # display the window infinitely until any keypress
cv2.destroyAllWindows()

img2 = img
img2[:, :, 0] = 0  # turn off the all blue pixels (BGR)
cv2.imwrite('./sample_out.jpg', img2)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()