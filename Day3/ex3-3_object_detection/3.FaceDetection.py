
"""
3. Face detection
"""

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt


Image = cv2.imread('./newjeans.jpg')

# if 'AttributeError: module 'cv2' has no attribute 'data' happens
# install opencv-contrib-python package
FaceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

Face = FaceDetector.detectMultiScale(Image)

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in Face])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(Image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# display the resulting frame

try:  # running in Pycharm
    cv2.imshow("Image", Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./newjeans_face_detection.png', Image)
except:  # running in Colab
    Image_rgb = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    plt.imshow(Image_rgb)
    plt.title("Image")
    plt.show()
    cv2.imwrite('./newjeans_face_detection.png', Image)

