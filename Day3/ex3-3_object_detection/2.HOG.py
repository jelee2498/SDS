
"""
Histogram of gradient (HOG)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


im = cv2.imread('./SDS/Day3/ex3-3_object_detection/people.jpg')
if isinstance(im, np.ndarray):  # running in Colab
    pass
else:  # running in Pycharm
    im = cv2.imread('./people.jpg')

# calculation of HoG feature quantity
hog = cv2.HOGDescriptor()

# create human identifier with HoG feature quantity + SVM
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# widStride :Window movement amount
# padding   :Extended range around the input image
# scale     :scale
hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}

# Detected person coordinate by the created identifier device
human, r = hog.detectMultiScale(im, **hogParams)

# Surround a person's area with a red rectangle
for (x, y, w, h) in human:
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 50, 255), 3)

# show image
try:  # running in Pycharm
  cv2.imshow("results human detect by DefaultHoGPeopleDetector", im)
  cv2.waitKey(0)
except:  # running in Colab
  im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  plt.imshow(im_rgb)
  plt.title("results human detect by DefaultHoGPeopleDetector")
  plt.show()