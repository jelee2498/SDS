""" Camera calibration (right) """
import numpy as np
import cv2
import glob



""" Calculate camera parameters """
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 5, 0)
objp = np.zeros((7*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./Calibration/R/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()



""" Calibration """
# mtx: camera matrix
# dist: distortion coefficients
# rvecs, tvecs: rotation, translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.save("./ret_R", ret)
np.save("./mtx_R", mtx)
np.save("./dist_R", dist)
np.save("./rvecs_R", rvecs)
np.save("./tvecs_R", tvecs)
np.save("./FocalLength_R", mtx[0,0]/mtx[1,1])



""" Remove distortion from a sample image """
img = cv2.imread('./Calibration/R/right01.png')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./Calibration_result_R.png', dst)
