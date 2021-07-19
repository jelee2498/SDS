
"""
Face detection in movie
"""

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt


movie = cv2.VideoCapture('./SDS/Day3/ex3-3_object_detection/AESPA.mp4')
out_path = './SDS/Day3/ex3-3_object_detection/AESPA_face_detection.avi'
if movie.isOpened():  # running in Colab
    pass
else:  # running in Pycharm
    movie = cv2.VideoCapture('./AESPA.mp4')
    out_path = './AESPA_face_detection.avi'

# define the codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, movie.get(cv2.CAP_PROP_FPS),
                      (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# load pretrained model
FaceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# detection parameter
Params = {'maxSize': (100, 100)}

# read until video is completed

while (movie.isOpened()):
    # capture frame-by-frame
    ret, frame = movie.read()

    if ret == True:

        # Detected person coordinate by the created identifier device
        Face = FaceDetector.detectMultiScale(frame, **Params)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in Face])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # display the resulting frame
        try:  # running in Pycharm
          cv2.imshow("Frame", frame)
        except:  # running in Colab
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          plt.imshow(frame_rgb)
          plt.title("Frame")
          plt.show()

        # write frame
        out.write(frame)

        # press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # break the loop
    else:
        break

# when everything done, release the video capture object
movie.release()

# closes all the frames
cv2.destroyAllWindows()