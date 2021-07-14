import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

movie = cv2.VideoCapture('./GFRIEND.mp4')
# check if camera opened successfully
if movie.isOpened() == False:
    print("Error opening video stream or file")

# define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_path = './test.avi'
out = cv2.VideoWriter('F:/MIPL/SDS/FaceDetection.avi', fourcc, movie.get(cv2.CAP_PROP_FPS),
                      (int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)), int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Load pretrained model
FaceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

# Detection parameter
Params = {'maxSize': (100, 100)}

# Read until video is completed
while (movie.isOpened()):
    # Capture frame-by-frame
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

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # write frame
        # out.write(frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
movie.release()

# Closes all the frames
cv2.destroyAllWindows()