import numpy as np

def normal_map(image, light, mask):

    nrows = image.shape[0] # get size of image
    ncols = image.shape[1]
    numimage = image.shape[2]

    surfnorm = np.zeros((nrows, ncols, 3)) # Initialize
    albedo = np.zeros((nrows, ncols))
    for i in range(0, nrows):
        for j in range(0, ncols):
            surfnorm[i, j, 0] = 0.0 # initialize normal(x)
            surfnorm[i, j, 1] = 0.0 # initialize normal(x)
            surfnorm[i, j, 2] = 1.0 # initialize normal(z)
            albedo[i, j] = 0.0 # initialize albedo

    for i in range(0, nrows):
        for j in range(0, ncols):
            if mask[i,j] > 0: # calculate only on the object area
                temp = np.zeros((numimage))
                for im in range(0, numimage):
                    temp[im] = np.double(image[i, j, im])
                NP, R, fail = pixelnorm(temp, light) # get pixel-wise outputs
                surfnorm[i, j, 0] = NP[0]
                surfnorm[i, j, 1] = NP[1]
                surfnorm[i, j, 2] = NP[2]
                albedo[i, j] = R

    maxval = np.max(albedo)
    if maxval > 0:
        albedo = albedo / maxval

    return surfnorm, albedo


def pixelnorm(image, light):

    # image.shape = 12      pixel values at a selected location per each image
    # light.shape = (12, 3) three light directions (x, y, z)

    # image = image.T

    fail = 0

    A = light.T.dot(light)
    b = light.T.dot(image)
    g = np.linalg.inv(A).dot(b)
    R = np.linalg.norm(g, ord=2) # Albedo
    N = np.divide(g, R) # Normal vector

    if np.linalg.norm(image, ord=2) < 1.0E-06:
        print('Warning: Pixel intensity is zero')
        N[0] = 0.0
        N[1] = 0.0
        N[2] = 0.0
        R    = 0.0
        fail = 1

    return N, R, fail

