""" Disparity map """
import cv2
import matplotlib.pyplot as plt



""" Load dataset """
img_L = cv2.imread('./Disparity/im0.png', cv2.IMREAD_GRAYSCALE)
img_R = cv2.imread('./Disparity/im1.png', cv2.IMREAD_GRAYSCALE)

plt.subplot(1,3,1), plt.imshow(img_L, 'gray')
plt.subplot(1,3,2), plt.imshow(img_R, 'gray')
plt.show()



""" Create disparity map """
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(img_L, img_R)

plt.subplot(1,3,3), plt.imshow(disparity)
plt.show()
