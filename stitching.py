import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.measure import block_reduce

"""File with all stitiching functions used for drone images"""

#------------------------------------------------#
"""xy stitching"""

def img_xymerge(df, xCoords, yCoords, xq, yq, pxSize,
                skip=1, imageType=False, coords = ['yc','xc'], method='nearest'):
    """Function loops over images to get xy grid of data points
    Input:
    df - dataframe with flight information
    xq, yq - image coordinate delimiters
    xCoords, yCoords - Coordinates relative to centre of each image for all pixels
    pxSize - pixel size of images
    skip - Use every nth image
    imageType - True if RGB image, False if numpy array type (converted thermal images)
    coords - name of coordinate columns to use from df
    method - interpolation method (nearest, linear, cubic)
    
    Output:
    xygrid - array of data values for each point on xy grid
    extent - Boundary coordinates for xy grid
    """
    xmCoords, ymCoords = np.meshgrid(xCoords[xq:3*xq],yCoords[yq:3*yq]) if xq is not False else np.meshgrid(xCoords,yCoords)
    xmcr, ymcr = xmCoords.reshape(-1,1), ymCoords.reshape(-1,1)

    alltot = [0,0,0,0,0] if imageType else [0,0,0]
    for index, row in tqdm(df[::skip].iterrows(), desc='Images processed'):
        imgst = hf.img_to_arr(row.imgPath, xq, yq)
        total = np.concatenate((xmcr+row[coords[0]], ymcr+row[coords[1]], imgst.reshape(-1,1) if len(imgst.shape) == 2 else [imgst[:,:,i].reshape(-1,1) for i in range(len(imgst.shape))]),axis=-1)
        alltot = np.vstack((alltot,total))

    alltot = alltot[1:,:]
    extent = (np.ceil(min(alltot[:,0])), np.floor(max(alltot[:,0])),np.ceil(min(alltot[:,1])),np.floor(max(alltot[:,1])))# minx, maxx, miny, maxy = min(total[:,0]), max(total[:,0]), min(total[:,1]), max(total[:,1])

    xsGrid = np.arange(extent[0], extent[1], pxSize)
    ysGrid = np.arange(extent[2], extent[3], pxSize)
    xsGridm, ysGridm = np.meshgrid(xsGrid, ysGrid)

    gridded = [griddata(alltot[1:,:2], alltot[1:,i], (xsGridm, ysGridm), method=method) for i in range(alltot.shape[1])[2:]]
    xygrid = gridded[0] if alltot.shape[1] == 3 else np.dstack(([gridded[i] for i in range(alltot.shape[1]-2)]))
    
    return xygrid, extent
  
#--------------------------------------#

"""RANSAC algorithm functions"""

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)
  
def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches
  
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None
      
#--------------------------------------------------#

"""Stitching stages functions"""

def initialise_vars(idf, start, xq, yq, tmin, tmax, path='imgPath', tempfiles=['temp0.jpg','temp1.jpg']):
    prev = start
    single = ((hf.img_to_arr(idf.iloc[start][path], xq=xq, yq=yq)-tmin)*255/tmax).astype(np.uint8)
    #imageio.imwrite(tempfiles[0],np.dstack((single,single,single)))
    prevImg = np.dstack((single,single,single))
    totalBox=[prevImg.shape[0], prevImg.shape[1]]
    prevBox=[0,prevImg.shape[0],0,prevImg.shape[1]]
    return prev, single, prevImg, totalBox, prevBox

def get_img_translation(trainImg, queryImg, feature_matching='bf', feature_extractor='orb'):
    # Opencv defines the color channel in the order BGR - transform it to RGB to be compatible to matplotlib
    trainImg_gray, queryImg_gray = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in [trainImg, queryImg]]

    # Detect the keypoints and features on both images
    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

    # Link the identified features between images
    if featuresA is None or featuresB is None: return None, None, None
    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)

    # Potential to improve by filtering out matches that are not in same direction of travel as drone
    ma = np.array([kpsA[j].pt for j in [i.queryIdx for i in matches]])
    mb = np.array([kpsB[j].pt for j in [i.trainIdx for i in matches]])
    
    diff = np.median(ma-mb, axis=0).astype(int)
    
    return ma, mb, diff

def filter_img_translation(ma, mb, df, num1, num2, movDef=2, x='xc', y='yc'):
    if ma is None or mb is None: return None, None, None
    # Filter by only matches in direction of travel
    xmov, ymov = df.iloc[num2][x]-df.iloc[num1][x], df.iloc[num2][y]-df.iloc[num1][y]
    if abs(xmov) < movDef/2 and abs(ymov) < movDef/2:
        mam, mbm = ma, mb
    else: 
        if abs(ymov) > abs(xmov):
            if ymov < 0:
                mam, mbm = ma[[ma[i,0]>mb[i,0] for i in range(len(ma))]], mb[[ma[i,0]>mb[i,0] for i in range(len(ma))]]
            else: mam, mbm = ma[[ma[i,0]>mb[i,0] for i in range(len(ma))]], mb[[ma[i,0]>mb[i,0] for i in range(len(ma))]]
        else:
            if xmov < 0:
                mam, mbm = ma[[ma[i,1]<mb[i,1] for i in range(len(ma))]], mb[[ma[i,1]<mb[i,1] for i in range(len(ma))]]
            else: mam, mbm = ma[[ma[i,1]>mb[i,1] for i in range(len(ma))]], mb[[ma[i,1]>mb[i,1] for i in range(len(ma))]]

    diff = np.median(mam-mbm, axis=0).astype(int)
    
    return mam, mbm, diff

def stitch_img_result(mam, mbm, diff, totalBox, prevBox, img_arrs, prevImg, prevNum, imgNum, min_matches=4, max_stdev=20, tmin=-10, tmax=40, verbose=True, queryImg=False):
  """Stitch individual images together"""
  
  if diff is None: 
      print('Images {} and {}, no matches'.format(str(prevNum),str(imgNum)))
      return totalBox, prevNum, prevImg, prevBox
  if verbose: print('Filt. matches: '+str(len(mam))+', stdev: ' + str(round(np.std(mam-mbm, axis=0).mean(),2)))

  # If many matches but too many outliers - take half around medians
  if len(mam) > min_matches and np.std(mam-mbm, axis=0).mean() > max_stdev:
      vals = np.mean(np.abs((mam-mbm)-np.median(mam-mbm,axis=0)),axis=1).argsort()[:int(len(mam)-5)]
      mam, mbm = mam[vals], mbm[vals]
      print('Removed outliers')

  # Filter for conditions
  if len(mam) > min_matches and np.std(mam-mbm, axis=0).mean() < max_stdev:
      # New box position before adjustment for expanding total box
      newBox=[int(np.round(prevBox[0]+diff[1])), int(np.round(prevBox[1]+diff[1])), int(np.round(prevBox[2]+diff[0])),int(np.round(prevBox[3]+diff[0]))] 
      pos = [0,0] # Position for previously merged images
      modBox = [0,0,0,0] # Position for new image

      # If bounds on axis 0 go beyond total
      if newBox[0]<0:
          xmin = imgNum
          modBox[1], modBox[0], pos[0] = newBox[1]-newBox[0], 0, abs(newBox[0])
          totalBox[0]+=abs(newBox[0])
      elif newBox[1] > totalBox[0]:
          xmax = imgNum
          modBox[1], modBox[0], pos[0] = newBox[1], newBox[0], 0
          totalBox[0]=newBox[1]
      else: modBox[0], modBox[1], pos[0] = newBox[0], newBox[1], 0

      # If bounds on axis 1 go beyond total
      if newBox[2]<0:
          ymin = imgNum
          modBox[3], modBox[2], pos[1] = newBox[3]-newBox[2], 0, abs(newBox[2])
          totalBox[1]+=abs(newBox[2])
      elif newBox[3] > totalBox[1]:
          ymax = imgNum
          modBox[3], modBox[2], pos[1] = newBox[3], newBox[2], 0
          totalBox[1] = newBox[3]
      else: modBox[2], modBox[3], pos[1] = newBox[2], newBox[3], 0    
      prevBox = modBox  

      if queryImg is False:
          single = ((img_arrs[1]-tmin)*255/tmax).astype(np.uint8)
          queryImg = np.dstack((single,single,single))
      result = np.zeros([totalBox[0],totalBox[1],3])
      result[pos[0]:pos[0]+prevImg.shape[0],pos[1]:pos[1]+prevImg.shape[1],:] = prevImg
      result[modBox[0]:modBox[1], modBox[2]:modBox[3],:] = queryImg
      print('Images {} and {} merged.'.format(str(prevNum),str(imgNum)))
      prevNum, prevImg = imgNum, result
  else: print('Images {} and {}, poor matching'.format(str(prevNum),str(imgNum)))
  return totalBox, prevNum, prevImg, prevBox
  
def stitch_img_result_pano(mam, mbm, diff, totalBox, prevBox, img_arrs, prevImg, prevNum, imgNum, min_matches=4, max_stdev=20, tmin=-10, tmax=40, verbose=True, rgb_query=False,inv=False):
  """Stitch strips of images together according to matching points"""
  
    if verbose: print('Filt. matches: '+str(len(mam))+', stdev: ' + str(round(np.std(mam-mbm, axis=0).mean(),2)))
    # Filter for conditions
    if len(mam) > min_matches and np.std(mam-mbm, axis=0).mean() < max_stdev:
        # New box position before adjustment for expanding total box
        if rgb_query is not False:
            newBox=[int(np.round(prevBox[0]+diff[1])), int(np.round(prevBox[0]+diff[1]))+rgb_query.shape[0], int(np.round(prevBox[2]+diff[0])),int(np.round(prevBox[2]+diff[0]))+rgb_query.shape[1]] 
        else: newBox=[int(np.round(prevBox[0]+diff[1])), int(np.round(prevBox[1]+diff[1])), int(np.round(prevBox[2]+diff[0])),int(np.round(prevBox[3]+diff[0]))] 
        pos = [0,0] # Position for previously merged images
        modBox = [0,0,0,0] # Position for new image
        # If bounds on axis 0 go beyond total
        if newBox[0]<0 and newBox[1] > totalBox[0]:
            xmin, xmax = imgNum, imgNum
            modBox[1], modBox[0], pos[0] = newBox[1]-newBox[0], 0, abs(newBox[0])
            totalBox[0]=newBox[1]-min(newBox[0],0)
        elif newBox[0]<0:
            xmin = imgNum
            modBox[1], modBox[0], pos[0] = newBox[1]-newBox[0], 0, abs(newBox[0])
            totalBox[0]+=abs(newBox[0])
        elif newBox[1] > totalBox[0]:
            xmax = imgNum
            modBox[1], modBox[0] = newBox[1], newBox[0]
            totalBox[0]=newBox[1]
        else: modBox[0], modBox[1] = newBox[0], newBox[1] 
                 
        # If bounds on axis 1 go beyond total
        if newBox[2]<0 and newBox[3] > totalBox[1]:
            ymin, ymax = imgNum, imgNum
            modBox[3], modBox[2], pos[1] = newBox[3]-newBox[2], 0, abs(newBox[2])
            totalBox[1]=newBox[3]-min(newBox[2],0)
        elif newBox[2]<0:
            ymin = imgNum
            modBox[3], modBox[2], pos[1] = newBox[3]-newBox[2], 0, abs(newBox[2])
            totalBox[1]+=abs(newBox[2])
        elif newBox[3] > totalBox[1]:
            ymax = imgNum
            modBox[3], modBox[2] = newBox[3], newBox[2]
            totalBox[1] = newBox[3] #-min(newBox[2],0)
        else: modBox[2], modBox[3] = newBox[2], newBox[3]
        prevBox = modBox 
        print(modBox)
        
        if len(img_arrs[1].shape) == 2:
            single = (img_arrs[1]-tmin)*255/tmax
            queryImg = np.dstack((single,single,single)).astype(np.uint8)
        else: queryImg = rgb_query
        result = np.zeros([totalBox[0],totalBox[1],3])
        if inv:
            result[modBox[0]:modBox[1], modBox[2]:modBox[3],:] = queryImg
            #prevImg.data[max(0,newBox[0]):min(newBox[1],prevImg.shape[0]), max(0,newBox[2]):min(newBox[3],prevImg.shape[1]),:] += np.array(queryImg[max(0,-newBox[0]):min(queryImg.shape[0],prevImg.shape[0]-max(newBox[0],0)), max(0,-newBox[2]):min(queryImg.shape[1],prevImg.shape[1]+max(-newBox[2],0)),:]*(prevImg.mask[max(0,newBox[0]):min(newBox[1],prevImg.shape[0]), max(0,newBox[2]):min(newBox[3],prevImg.shape[1]),:]))
            prevImg.data[max(newBox[0],0):min(newBox[1],prevImg.shape[0]), max(newBox[2],0):min(newBox[3],prevImg.shape[1]),:] += np.array(queryImg[-min(newBox[0],0):min(queryImg.shape[0],prevImg.shape[0]-newBox[0]), -min(newBox[2],0):min(queryImg.shape[1],prevImg.shape[1]-newBox[2]),:]*prevImg.mask[max(newBox[0],0):min(newBox[1],prevImg.shape[0]), max(newBox[2],0):min(newBox[3],prevImg.shape[1]),:])
            result[pos[0]:pos[0]+prevImg.shape[0],pos[1]:pos[1]+prevImg.shape[1],:] = prevImg   
        else:    
            result[pos[0]:pos[0]+prevImg.shape[0],pos[1]:pos[1]+prevImg.shape[1],:] = prevImg
            print(-min(newBox[0],0),min(queryImg.shape[0],prevImg.shape[0]-newBox[0]), -min(newBox[2],0),min(queryImg.shape[1],prevImg.shape[1]-newBox[2]))
            queryImg.data[-min(newBox[0],0):min(queryImg.shape[0],prevImg.shape[0]-newBox[0]), -min(newBox[2],0):min(queryImg.shape[1],prevImg.shape[1]-newBox[2])] += np.array(prevImg[max(0,newBox[0]):min(prevImg.shape[0],newBox[1]), max(0,newBox[2]):min(prevImg.shape[1],newBox[3]),:]*(queryImg.mask[-min(newBox[0],0):min(queryImg.shape[0],prevImg.shape[0]-newBox[0]), -min(newBox[2],0):min(queryImg.shape[1],prevImg.shape[1]-newBox[2])]))
            result[modBox[0]:modBox[1], modBox[2]:modBox[3],:] = queryImg
        print('Images {} and {} merged.'.format(str(prevNum),str(imgNum)))
        prevNum, prevImg = imgNum, result
    else: print('Images {} and {}, poor matching'.format(str(prevNum),str(imgNum)))
    return totalBox, prevNum, prevImg, prevBox