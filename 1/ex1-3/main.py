""" Photometric stereo """
import cv2
import scipy.io
from normal_map import normal_map



""" Load dataset """
I = scipy.io.loadmat('cat_image.mat')['I']
mask = scipy.io.loadmat('cat_mask.mat')['mask']
L = scipy.io.loadmat('cat_light.mat')['L']



""" Show images """
cv2.imshow('test', I[:,:,0])
cv2.waitKey(0)
cv2.destroyAllWindows()



""" Calculate normal map """
N, AL = normal_map(I, L, mask)
b, g, r = cv2.split(N)   # img to bgr
N = cv2.merge([r, g, b]) # bgr to rgb and merge



""" Check outputs """
cv2.imshow('N', N)
cv2.imshow('AL', AL)
cv2.waitKey(0)
cv2.destroyAllWindows()
