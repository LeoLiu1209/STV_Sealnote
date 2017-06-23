import numpy as np
import cv2
    
def assertVisa(fileName):
    cardType='visa.png'
    img1 = cv2.imread(cardType,0)      # queryImage
    screenShot = cv2.imread(fileName)  # trainImage
    chopScreenShot=cv2.resize(screenShot,(screenShot.shape[1]/2,screenShot.shape[0]/2))
          

    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(chopScreenShot,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    count=0
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            count=count+1
            matchesMask[i]=[1,0]
    if count>15:
        return True
    else:
        return False



def assertMaster(fileName):
    cardType='master.png'     
    img1 = cv2.imread(cardType,0)      # queryImage
    screenShot = cv2.imread(fileName)  # trainImage
    chopScreenShot=cv2.resize(screenShot,(screenShot.shape[1]/2,screenShot.shape[0]/2))
          

    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(chopScreenShot,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    count=0
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            count=count+1
            matchesMask[i]=[1,0]
    if count>15:
        return True
    else:
        return False


def assertExpress(fileName):
    cardType='express.png'     
    img1 = cv2.imread(cardType,0)      # queryImage
    screenShot = cv2.imread(fileName)  # trainImage
    chopScreenShot=cv2.resize(screenShot,(screenShot.shape[1]/2,screenShot.shape[0]/2))
          

    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=100)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(chopScreenShot,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    count=0
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            count=count+1
            matchesMask[i]=[1,0]
    if count>15:
        return True
    else:
        return False

