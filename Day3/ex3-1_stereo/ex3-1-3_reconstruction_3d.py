""" 3d reconstruction """
import numpy as np
import cv2
import matplotlib.pyplot as plt



""" Initialize """
# Read parameters
ret_L = np.load('./ret_L.npy')
ret_R = np.load('./ret_R.npy')
mtx_L = np.load('./mtx_L.npy')
mtx_R = np.load('./mtx_R.npy')
dist_L = np.load('./dist_L.npy')
dist_R = np.load('./dist_R.npy')

# Read image sample
img_L = cv2.imread('./Calibration/L/left01.png')
img_R = cv2.imread('./Calibration/R/right01.png')



""" Remove distortion """
# Get height and width
h, w = img_L.shape[:2]

# Get optimal camera matrix for better undistortion
new_camera_matrix_L, roi_L = cv2.getOptimalNewCameraMatrix(mtx_L, dist_L, (w, h), 1, (w, h))
new_camera_matrix_R, roi_R = cv2.getOptimalNewCameraMatrix(mtx_R, dist_R, (w, h), 1, (w, h))

# Undistort images
img_L_undistorted = cv2.undistort(img_L, mtx_L, dist_L, None, new_camera_matrix_L) # Undistort
img_R_undistorted = cv2.undistort(img_R, mtx_R, dist_R, None, new_camera_matrix_R)
x_L, y_L, w_L, h_L = roi_L # Crop
x_R, y_R, w_R, h_R = roi_R
img_L_undistorted = img_L_undistorted[y_L:y_L+h_L, x_L:x_L+w_L]
img_R_undistorted = img_R_undistorted[y_R:y_R+h_R, x_R:x_R+w_R]
img_L_undistorted = cv2.resize(img_L_undistorted, (w,h))
img_R_undistorted = cv2.resize(img_R_undistorted, (w,h))
img_L_undistorted = cv2.resize(img_L_undistorted, None, fx=0.5, fy=0.5)
img_R_undistorted = cv2.resize(img_R_undistorted, None, fx=0.5, fy=0.5)



""" Calculate disparity map """
# Set disparity parameters
win_size = 7
min_disp = -1
max_disp = 63
num_disp = max_disp - min_disp

# Create Block matching object
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=3,
                               uniquenessRatio=0,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=32,
                               P1=8*3*win_size**2,
                               P2=32*3*win_size**2)

# Compute disparity map
disparity_map = stereo.compute(img_L_undistorted, img_R_undistorted)

# Show disparity map before generating 3D cloud to verify that point cloud will be usable
plt.imshow(disparity_map)
plt.show()



""" 3D reconstruction based on the disparity map """
#Load focal length.
focal_length_L = np.load('./FocalLength_L.npy')

#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q = np.float32([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, focal_length_L*0.5, 0],#Focal length multiplication obtained experimentally.
                [0, 0, 0, 1]])

#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)

#Get color points
colors = cv2.cvtColor(img_L_undistorted, cv2.COLOR_BGR2RGB)

#Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()



""" Save outputs as mesh file """
# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

#Generate point cloud
print ("\n Creating the output file... \n")
output_points = np.hstack([output_points.reshape(-1,3),output_colors])
output_colors = output_colors.reshape(-1,3)
output_file = './reconstructed.ply'
ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
with open(output_file, 'w') as f:
    f.write(ply_header %dict(vert_num=len(output_points)))
    np.savetxt(f, output_points,'%f %f %f %d %d %d')



""" Run meshlab """
