
"""
Histogram of gradient (HOG)
"""

import cv2


im = cv2.imread('./SDS/Day3/ex3-3_object_detection/people.jpg')
if im:  # running in Colab
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
cv2.imshow("results human detect by DefaultHoGPeopleDetector", im)
cv2.waitKey(0)