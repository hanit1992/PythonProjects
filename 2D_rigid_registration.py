# in this exercise we will find a rigid registration between 2 2D images. we will find the registration with outliers,
# without outliers, when performing the process once and twice - and comparing all the results.

import numpy as np
from scipy import ndimage

import utils
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage import transform,io
from PIL import Image

FEATURE_POINTS_COLOR = 'c'
PAIR_NUMBER_COLOR = 'blue'
WITH_OUTLIERS = 'with_outliers'
NO_OUTLIERS = 'no_outliers'
ANNOTATE_LOCATION_ADDITION = 15
PAIR_GRAPH_BL_HEADLINE = 'BL_image with feature point'
PAIR_GRAPH_FU_HEADLINE = 'FU_image with feature point'

# 1. in the first part we will load 2 2D images, and call a function that will find us pairs of feature regions on both
#    images, that should match each other. in this part, the points we will get would be with no outliers. we will then
#    plot the points to a graph, view the pairs - and analyse them.

def getPointsAsMatrices(BLPoints,FUPoints):
    return np.transpose(BLPoints),np.transpose((FUPoints))

"""

"""
def add_image_figure(image,points,num,title):
    fig = plt.figure(num)
    fig.suptitle(title)
    plt.imshow(image)
    x_coords, y_coords = zip(*points)
    plt.scatter(x=x_coords, y=y_coords, c=FEATURE_POINTS_COLOR)
    # lets add some annotation to the points pair number - it's the same as the array index
    for i in range(len(points)):
        plt.annotate(str(i+1),xy = points[i],xytext = (points[i][0]+ANNOTATE_LOCATION_ADDITION,
                                                       points[i][1]+ANNOTATE_LOCATION_ADDITION),
                                                       color = PAIR_NUMBER_COLOR)

"""
this function will load 2 images, and will subplot the points matching. it will marks the pairs by numbers
"""
def present_images_feature_pairs(BL_image, FU_image):
    # lets get the data from the given files
    BL_image, FU_image = readImages(BL_image, FU_image)
    # lets get the pair of points, but with no outliers
    BLPoints, FUPoints = utils.getPoints(NO_OUTLIERS)

    # and now we will plot the pair of points on top of the two images
    add_image_figure(FU_image, FUPoints,1,PAIR_GRAPH_FU_HEADLINE)
    add_image_figure(BL_image,BLPoints,2,PAIR_GRAPH_BL_HEADLINE)
    # show me the graph created
    plt.show()

def readImages(BL_image, FU_image):

    # img = io.imread(FU_image,as_grey=True)
    BL_image = plt.imread(BL_image)
    FU_image = plt.imread(FU_image) # note: this images ar "rgb" - they have 3 equivelent chennels

    # # lets get rid of other chanels. tiff images has 3 channels with same values
    # BL_image = BL_image[:,:,0]
    # FU_image = FU_image[:,:,0]

    return BL_image, FU_image

# 2. in this part we will calculate the registration matrix - given a set of pair points that we count on (meaning -
#    no outliers) - we will use SVD to get the matrix. the matrix will be 3x3, meaning that we would have to activate
#    it on the 2d points with homogney system (so they will be 3 coordinate points)

"""
this function will return an ndArray object which is the registration matrix
"""
def calcPointBasedReg(BLPoints, FUPoints):
    # our goal would be to minimize the MSE - finding the Matrix who will minimize it. we will do so in 4 stages

    # 1. calculate the weighted centroid of both point sets
    BL_centroid, FU_centroid = calcCentroidsOfBothSets(BLPoints,FUPoints)

    # 2. lets compute the centered vectors - meaning we will substruct the centroid from each point
    BLMatrixCenteredPoints, FUMatrixCenteredPoints = calcCentroidPoints(BLPoints,FUPoints,BL_centroid,FU_centroid)

    # 3. lets get the covariance S matrix
    S = calcCovarianceMatrix(BLMatrixCenteredPoints, FUMatrixCenteredPoints)

    # 4. lets decompose the matrix S to SVD components
    U,V_T = decomposeMatrixBySVD(S)

    # 5. now, we can get R - the Rotation matrix, from the results
    R = getRotationMatrix(U,V_T)

    # R = getRotationMatrix2(U, V_T).transpose()

    # 6. finally, lets get the translation vector

    t = BL_centroid - np.dot(R.transpose(),FU_centroid)
    #
    # t = FU_centroid - np.dot(R.transpose(), BL_centroid)

    # now we would like to calculate rigidReg as a homogeneous matrix
    Reg = getHomogeneousFullMatrix2(R,t)
    # Reg = transform.AffineTransform(Reg)
    print(Reg)
    # calcDist(BLPoints,FUPoints,Reg)
    return Reg

# todo this is the one I use now
def getHomogeneousFullMatrix2(R,t):
    # so we need to create one 3x3 matrix
    final = np.zeros((3,3))
    final[0:2, 0:2] = R
    # newR = np.zeros((3,2))
    # newR[0:2,0:2] = R
    t = np.append(t,np.array([1]))
    # Reg = np.zeros((3,3))
    # Reg[:,0:2] = newR

    # we need to switch the values of t. the reason is difference in coordinate systems
    final[2] = t
    return final

def getHomogeneousFullMatrix(R,t):
    # so we need to create one 3x3 matrix
    newR = np.zeros((3,2))
    newR[0:2,0:2] = R
    t = np.append(t,np.array([1]))
    Reg = np.zeros((3,3))
    Reg[:,0:2] = newR
    Reg[:,2] = t
    return Reg

def getRotationMatrix2(U,V_T):
    # U_transpose = np.transpose(U)
    # V = np.transpose(V_T)
    V_UT = np.dot(U,V_T).transpose()
    centeralMatrix = np.identity(U.shape[0])
    last_column = np.zeros(U.shape[1])
    last_column[-1] = np.linalg.det(V_UT)
    centeralMatrix[-1] = last_column

    VC = np.matmul(U, centeralMatrix)
    R = np.dot(VC, V_T)

    # return np.transpose(R)
    return R

def getRotationMatrix(U,V_T):
    U_transpose = np.transpose(U)
    V = np.transpose(V_T)
    centeralMatrix = np.identity(U.shape[0])
    last_column = np.zeros(U.shape[1])
    last_column[-1] = np.linalg.det(np.matmul(V,U_transpose))
    centeralMatrix[-1] = last_column

    VC = np.matmul(V,centeralMatrix)
    R = np.matmul(VC,U_transpose)

    # return np.transpose(R)
    return R

def decomposeMatrixBySVD(S):
    U,D,V_T = np.linalg.svd(S,full_matrices=True)
    # print(np.allclose(S,np.matmul(np.matmul(U,(np.diag(D))),V_T)))
    # return U,np.transpose(V_T)
    return U, V_T

"""
the covariance matrix S will be calculated by multiplying both BL point and FU points matrix, along with the unit matrix
(it's the unit matrix because the weights are 1)
"""
def calcCovarianceMatrix(BLCenteredPoints, FUCenteredPoints):

    X = FUCenteredPoints
    Y_trans = np.transpose(BLCenteredPoints)
    # X = BLCenteredPoints
    # Y_trans = np.transpose(FUCenteredPoints)

    # taking the identity matrix with the size of number of points
    # W = np.identity(BLCenteredPoints.shape[1])

    ones = np.ones((len(Y_trans)))
    W = np.diag(ones)


    # multiplying the 2 first matrix
    # XW = np.dot(X,W)
    # # multiplying the result by a third one, getting the covariance
    # S = np.dot(XW,Y_trans)
    S = np.dot(X,Y_trans)

    return S

"""
rigidReg is an homoganey 3x3 matrix
"""
def calcDist(BLPoints,FUPoints,rigidReg):

    # lets first get all the data which we want to measure it's distance
    homoganyBLPoints,homoganyFUPoints = getHomogenyPoints(BLPoints,FUPoints)
    transformedFUPoints = np.dot(homoganyFUPoints,rigidReg)
    # transformedBLPoints = np.dot(homoganyBLPoints, rigidReg)
    # now we can use euclidean distance in order to measure what we want
    distVector = np.sqrt((transformedFUPoints[:,0]-homoganyBLPoints[:,0])**2 +
                         (transformedFUPoints[:,1]-homoganyBLPoints[:,1])**2)

    # we will get the RMSE also
    RMSE = calcRMSE(transformedFUPoints,homoganyBLPoints)
    print(RMSE)
    return distVector

"""

"""
def calcRMSE(transformedFUPoints,homoganyBLPoints):
    rmse = np.sqrt((mean_squared_error(transformedFUPoints,homoganyBLPoints)))
    return rmse

"""

"""
def getRMSE(distVector):
    rmse = np.sqrt(((distVector) ** 2).mean())
    return rmse

"""

"""
def calcCentroidPoints(BLPoints,FUPoints,BL_centroid,FU_centroid):
    BLCenteredPoints = BLPoints-BL_centroid
    FUCenteredPoints = FUPoints-FU_centroid
    return getPointsAsMatrices(BLCenteredPoints,FUCenteredPoints)

def calcCentroidsOfBothSets(BLPoints, FUPoints):
    # we are giving all the points the same weight, so it's 1. so it;s basically just the mean of the points
    BLCentroid = sum(BLPoints)/len(BLPoints)
    FUCentroid = sum(FUPoints)/len(FUPoints)

    return BLCentroid,FUCentroid

"""

"""
def getHomogenyPoints(BLPoints, FUPoints):
    ones = np.ones((len(BLPoints), 1))
    homoganyFUPoints = np.append(FUPoints, ones, axis=1)
    homoganyBLPoints = np.append(BLPoints, ones, axis=1)
    return homoganyBLPoints,homoganyFUPoints

def getAlignedImages():

    # first, we will load the 2 images we would like to align
    BL_image, FU_image = readImages("BL01.tif","FU01.tif")

    # now lets get the feature points - no outliers
    BLPoints, FUPoints = utils.getPoints(NO_OUTLIERS)

    # lets get the registration, to activate on FUPoints
    reg = calcPointBasedReg(BLPoints,FUPoints)
    print(reg)
    # reg[2] = -reg[2]

    # registered_FU_img = ndimage.affine_transform(FU_image[:, :, 0], reg[0:2], -reg[2][::-1])
    # registered_FU_img = np.tile(registered_FU_img.reshape(registered_FU_img.shape + (1,)), 3)
    # registered_FU_img[:, :, 0] = 255
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(BL_image)
    # ax.imshow(registered_FU_img, alpha=0.5)
    # plt.show()

    # presentAlignedImages(BL_image,registered_FU_img)

    # now lets perform the registration on the FU_image

    # first we want to get another representation for the matrix
    registrationMatrix = transform.AffineTransform(matrix=np.transpose(reg))
    # now lets warp the FU_Image to align with BL_Image
    FU_Transformed = transform.warp(FU_image,registrationMatrix)

    presentAlignedImages(BL_image,FU_Transformed)

    return

def presentAlignedImages(BL_image, FU_Transformed):
    # lets change the color of one of the images to see difference
    BL_image_copy = np.copy(BL_image)
    BL_image_copy[:, :, 0] = 255
    im1 = Image.fromarray(BL_image_copy)
    im2 = Image.fromarray((FU_Transformed * 255).astype(np.uint8))
    result = Image.blend(im2, im1, 0.5)
    plt.imshow(result)
    plt.show()


# calling functions and testing

# test present points function
# present_images_feature_pairs("BL01.tif","FU01.tif")

# # test registration calculation and error
# BLPoints, FUPoints = utils.getPoints(NO_OUTLIERS)
# Reg = calcPointBasedReg(BLPoints,FUPoints)
# errorVec = calcDist(BLPoints,FUPoints,Reg)
# print(errorVec)

# test alignment and presentation:
getAlignedImages()
