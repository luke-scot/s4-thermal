import numpy as np
import glob
import re
import cv2
import imageio
#import rasterio as ro
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from pyproj import Proj
from skimage.measure import block_reduce

#-----------------------------------------------#
def img_info_merge(imgDir, pathFile, utcDiff, pathColumns, imageType=False, corr=True):
    """Function takes image directory and a flight path .csv and merges information for each
    image according to the file name of the image which denotes a timestamp
    
    Input:
    imgDir - Directory name ending with "/" containing image files
    pathFile - File to .csv flight path from litchi application
    utcDiff - Time difference of flight info vs UTC
    imageType - True if RGB image, False if numpy array type (converted thermal images)
    
    Output:
    merged - Pandas dataframe with entry for every image containing associated flight information
    """
    
    ## Get image dataframe with corresponding properties extracted frpm path file
    fileTypes = ('.jpg','.png','.tif') if imageType else ('.npy')
    imgs = [_ for _ in glob.glob(imgDir+'*.*') if _.endswith(fileTypes)]
    imgs.sort()
    # Extract date and time from filenames
    imgs = [i.replace('\\','/') for i in imgs]
    imgdates = [re.search('/20(.+?)_', path).group(1) for path in imgs] # Extract date from filename
    imgtimes = [re.search(imgdates[i]+'_(.+?)_', imgs[i]).group(1) for i in range(len(imgs))] # Extract time from filename
    # Convert to unix datetime 
    imgdatetimes = np.array([(datetime.timestamp(datetime(int('20'+imgdates[i][:2]),int(imgdates[i][2:4]),int(imgdates[i][4:6]),int(imgtimes[i][:2])+utcDiff,int(imgtimes[i][2:4]),int(imgtimes[i][4:6])))) for i in range(len(imgs))])*1000
    
    # Imprt paths and get corresponding timestamps for images
    pathDf = pd.read_csv(pathFile)
    if corr is True: corr = pathDf[pathDf['isflying']==1].iloc[-1].timestamp-imgdatetimes[-1]
    elif corr is False: corr=0
    
    # Get nearest GPS timestamp
    gpstimes = [min(pathDf['timestamp'], key=(lambda list_value : abs(list_value - i))) for i in imgdatetimes+corr]
    
    # Create image dataframe
    imgDf = pd.DataFrame(data=np.array([imgs,gpstimes]).transpose(),columns=['imgPath','timestamp'])
    imgDf['timestamp'] = imgDf['timestamp'].astype(float)

    # Merge with path dataframe
    merged = imgDf.merge(pathDf[pathColumns], on='timestamp', how='left')
    
    return merged

def filter_imgs(df, properties = [], values = []):
    """Filters pandas dataframe according to properties and a range of values
    
    Input:
    df - Pandas dataframe
    properties - Array of column names to be filtered
    values - Array of tuples containing bounds for each filter
    
    Output:
    df - Filtered dataframe
    """
    for i, val in enumerate(properties):
        df = df.loc[(df[val] > values[i][0]) & (df[val] < values[i][1])]
    return df


def reproject_coords(df, utmZone, hemisphere='north', inverse=False, 
                     orig=['longitude','latitude'], local=['x','y']):
    """Reprojects coordinates in dataframe to new column
    
    Input:
    df - Dataframe containg coordinates
    utmZone - WGS84 UTM zone
    hemisphere - north or south
    inverse - False for lat/lon to local, True for local to lat/lon
    
    Output:
    df - Dataframe with new column for coordinates
    """
    # Convert coordinates to UTM
    myProj = Proj('+proj=utm +zone='+utmZone+', +'+hemisphere+' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    if inverse: df[orig[0]], df[orig[1]] = myProj(df[local[0]].values, df[local[1]].values, inverse=True)
    else: df[local[0]], df[local[1]] = myProj(df[orig[0]].values, df[orig[1]].values)
    return df, myProj
  
def correct_coords(df, distFilt=False, altitude='altitude(m)', yaw='yaw(deg)',
                   pitch='pitch(deg)', roll='roll(deg)', degs=True,
                   ins=['x','y'], outs=['xc','yc']):
    """Converts aerial coordinates to ground coordinates for an image by taking
    into account pitch and roll of aircraft
    
    Input:
    df - Dataframe containing UTM coordinates
    distFilt - Filter out rows where distance between ground and aerial coord is over value
    altitude, yaw, pitch, roll - Dataframe columns for values
    degs - Values in degrees, False in radians
    ins, outs - input and output column names
    
    Output:
    df - Dataframe containing columns for converted coordinates
    """
    if degs: yaw, pitch, roll = [np.deg2rad(df[i]) for i in [yaw, pitch, roll]]
    else: yaw, pitch, roll = [df[i] for i in [yaw, pitch, roll]]
    # Pitch & Roll distances
    dist, dist2 = df[altitude]*np.tan(pitch), df[altitude]*np.tan(roll)
    # Correct x and y
    df[outs[0]], df[outs[1]] = df[ins[0]]+(dist*np.sin(yaw))+(dist2*np.cos(yaw)), df[ins[1]]+(dist*np.cos(yaw))+(dist2*np.sin(yaw))
    
    # Apply distance filter if True
    return df[abs(np.hypot(dist,dist2)) < distFilt] if distFilt else df

  #-------------------------------------------------------#

"""Import functions"""
def img_to_arr(filepath, xq=False, yq=False):
    if '.npy' in filepath: 
        arr = np.load(filepath)
        if xq and yq: arr = arr[yq:arr.shape[0]-yq,xq:arr.shape[1]-xq]
    else:
        img = imageio.imread(filepath)
        #img = ro.open(filepath)
        read = img.read()[:, yq:img.shape[0]-yq,xq:img.shape[1]-xq] if xq and yq else img.read()
        arr = np.dstack((read[0],read[1],read[2]))/255  
    return arr
  
# Function downsamples an image input as an array
def downsample_arr(arr, pxSize, resolution, sampleType=np.mean):
    ds = int(np.floor(resolution/pxSize))
    if len(arr.shape) == 3:       
        return np.dstack(([block_reduce(arr[:-(arr.shape[0] % ds),:-(arr.shape[1] % ds), i], (ds, ds), sampleType, cval = arr.mean()) for i in range(arr.shape[2])]))
    else: 
        return block_reduce(arr[:-(arr.shape[0] % ds),:-(arr.shape[1] % ds)], (ds, ds), sampleType, cval = arr.mean())
      
def use_centre(useCentre, df, pxSize, path='imgPath'):
    """Fetch relative coordinates for only using centre of images if this is specified in variables.
    
    Input:
    useCentre - boolean for using centre of images
    df - dataframe with flight information
    pxSize - pixel size of images
    path - column containing image path in df
    
    Output:
    xq, yq - image coordinate delimiters
    xCoords, yCoords - Coordinates relative to centre of each image for all pixels
    """
    arr = img_to_arr(df.iloc[0][path])
    yCoords, xCoords = [(np.array(range(arr.shape[i]))-(arr.shape[i]/2))*pxSize+pxSize/2 for i in [0,1]]
    if useCentre: xq, yq = int(np.floor(len(xCoords)/4)), int(np.floor(len(yCoords)/4))
    else: xq, yq = False, False
    return xq, yq, xCoords, yCoords