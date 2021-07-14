
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

#-1 Feature matching

# # cv2.imread(fileName, flag)
# # flag option#1 :cv2.IMREAD_COLOR (1)
# # flag option#2 :cv2.IMREAD_GRAYSCALE (0)
# # flag option#3 :cv2.IMREAD_UNCHANGED (-1)
# Image = cv2.imread('./mingky.jpg', 0)
# objectImage = cv2.imread('./mingkyDoll.jpg', 0)
#
# # calculate keypoints
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(objectImage, None)
# kp2, des2 = orb.detectAndCompute(Image, None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
# matches = bf.match(des1, des2)
#
# # sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)
# good_matches = matches[:6]
#
# img3 = cv2.drawMatches(objectImage, kp1, Image, kp2, good_matches, None, flags=2)
#
# plt.figure(dpi=150)
# plt.imshow(img3)
#
# plt.show()


#-2 HOG

# im = cv2.imread('./people.jpg')
#
# # calculation of HoG feature quantity
# hog = cv2.HOGDescriptor()
#
# # create human identifier with HoG feature quantity + SVM
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# # widStride :Window movement amount
# # padding   :Extended range around the input image
# # scale     :scale
# hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
#
# # Detected person coordinate by the created identifier device
# human, r = hog.detectMultiScale(im, **hogParams)
#
# # Surround a person's area with a red rectangle
# for (x, y, w, h) in human:
#     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 50, 255), 3)
#
# # show image
# cv2.imshow("results human detect by DefaultHoGPeopleDetector", im)
# cv2.waitKey(0)


#-3 Face detection

# Image = cv2.imread('./GFRIEND.jpg')
#
# # if 'AttributeError: module 'cv2' has no attribute 'data' happens
# # install opencv-contrib-python package
# FaceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# # FaceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# Face = FaceDetector.detectMultiScale(Image)
#
# # apply non-maxima suppression to the bounding boxes using a
# # fairly large overlap threshold to try to maintain overlapping
# # boxes that are still people
# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in Face])
# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
#
# # draw the final bounding boxes
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(Image, (xA, yA), (xB, yB), (0, 255, 0), 2)
#
#
# # Display the resulting frame
# cv2.imshow('Image', Image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imwrite('./FaceDetectionResult.png', Image)


#-4 Face detection movie

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