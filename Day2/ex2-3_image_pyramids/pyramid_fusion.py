""" Image pyramid """
import cv2
import matplotlib.pyplot as plt
import numpy as np

A = cv2.imread('./tony.png', cv2.IMREAD_GRAYSCALE)
B = cv2.imread('./iron.png', cv2.IMREAD_GRAYSCALE)

""" for image A """
G = A.copy()
gpA = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpA.append(G)

lpA = [gpA[4]]
for i in range(4,0,-1):
    GE = cv2.pyrUp(gpA[i])
    temp = cv2.resize(gpA[i-1], (GE.shape[:2][1], GE.shape[:2][0]))
    L = cv2.subtract(temp,GE)
    lpA.append(L)

""" for image B """
G = B.copy()
gpB = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpB.append(G)

lpB = [gpB[4]]
for i in range(4,0,-1):
    GE = cv2.pyrUp(gpB[i])
    temp = cv2.resize(gpB[i - 1], (GE.shape[:2][1], GE.shape[:2][0]))
    L = cv2.subtract(temp, GE)
    lpB.append(L)

""" fusing """
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols = la.shape
    ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1,5):
    ls_ = cv2.pyrUp(ls_)
    temp = cv2.resize(LS[i],(ls_.shape[:2][1], ls_.shape[:2][0]))
    ls_ = cv2.add(ls_, temp)

""" hard concatenating """
real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))

cv2.imshow('lap', ls_)
cv2.imshow('real', real)

cv2.destroyAllWindows()