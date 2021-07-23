import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper_functions as hf
from collections import Counter
from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
                       AltitudeMode, Camera)

#--------------------------------------------#
"""Matplotlib plotting functions"""

def plot_orientation(fig, ax, df, pitch='pitch(deg)', roll='roll(deg)', yaw='yaw(deg)'):
    """Plot pitch/roll/yaw of aircraft over time
    
    Input:
    fig - figure for plot
    ax - subplot in figure
    df - dataframe with flight information
    pitch, roll, yaw - Column names
    
    Output:
    fig - updated figure
    ax - updated subplot
    """
    ax.plot(df.index,df[pitch],label='pitch')
    ax.plot(df.index,df[roll],label='roll')
    par1 = ax.twinx()
    par1.plot(df.index,df[yaw],c='g',label='yaw')
    ax.set_title('Pitch/roll/yaw evolution')
    ax.set_xlabel('Image number'), ax.set_ylabel('Degrees (pitch & roll)')
    par1.set_ylabel('Degrees (yaw)')
    fig.legend(loc='upper left')
    return fig, ax

def plot_route(fig, ax, df, xq, yq, pxSize, skip=15, coords=['xc', 'yc'], path='imgPath'):
    """Plot locations of images used and example extents
    Input:
    fig - figure for plot
    ax - subplot in figure
    df - dataframe with flight information
    xq, yq - image coordinate delimiters from usecentre()
    pxSize - pixel size of images
    skip - plot extent for every n images
    coords - names of columns with coordinate information
    path - column containing image path in df
    
    Output:
    fig - updated figure
    ax - updated subplot
    """
    b = ax.scatter(df[coords[0]], df[coords[1]], c=df.index, cmap='Spectral')
    arr = hf.img_to_arr(df.iloc[0][path], xq=xq, yq=yq)
    size = np.array(arr.shape[:2])*pxSize/2
    for i, row in df[::skip].iterrows():
        ax.plot([row[coords[0]]-size[1], row[coords[0]]+size[1]], [row[coords[1]], row[coords[1]]],'k-o')
        ax.plot([row[coords[0]], row[coords[0]]],[row[coords[1]]-size[1], row[coords[1]]+size[1]],'k-o')
    ax.set_title('Used image coordinates')
    ax.legend(labels=['Extent for every {}th'.format(skip)])
    fig.colorbar(b, ax=ax,label='Image number')
    return fig, ax
  
def plot_image(fig, ax, df, xq, yq, pxSize, resolution, cmap='hot', imageNum=False, imageType=False, path='imgPath'):
    """Plot image from dataframe
    Input:
    fig - figure for plot
    ax - subplot in figure
    df - dataframe with flight information
    xq, yq - image coordinate delimiters from usecentre()
    pxSize - pixel size of images (m)
    resolution - desired resolution of image (m)
    cmap - colormap for plot (only used if 1D image, not RGB)
    imageNum - Number of image to plot
    imageType - True if RGB, False if 1D
    path - column containing image path in df
    
    Output:
    fig - updated figure
    ax - updated subplot
    """
    if imageNum is False: imageNum = int(len(df.index)/2)        
    arr = hf.img_to_arr(df.iloc[imageNum][path], xq=xq, yq=yq)
    ds_array = hf.downsample_arr(arr, pxSize, resolution) if pxSize != resolution else arr
    c = ax.imshow(ds_array) if imageType else ax.imshow(ds_array, vmin=arr.min(), vmax=arr.max(),cmap=cmap)
    if pxSize == resolution: ax.set_title('Image {}'.format(imageNum))
    else: ax.set_title('Image {} downsampled to {} m resolution'.format(str(imageNum),resolution))
    if imageType is False: fig.colorbar(c, ax=ax, label=' Temperature ($^{\circ}$C)')
    return fig, ax

  
def plot_array(fig, ax, arr, extent, cmap='hot', title=False, imageType=False, scale=False):
    """Plot array as an image
    Input:
    fig - figure for plot
    ax - subplot in figure
    arr - array to be plotted
    extent - Spatial extent of plotted image
    cmap - colormap for plot (only used if 1D image, not RGB)
    title - Optional title for figure
    imageType - True if RGB, False if 1D
    scale - scaling for cmap in 1D case
    
    Output:
    fig - updated figure
    ax - updated subplot    
    """
    if imageType: ax.imshow(arr,extent=extent)
    else:
        if scale is not False: a = ax.imshow(arr,extent=extent,cmap=cmap,vmin=scale.min(),vmax=scale.max())  
        else: a = ax.imshow(arr,extent=extent,cmap=cmap)
        c = fig.colorbar(a, ax=ax)
        c.set_label('Temperature ($^{\circ}$C)')
    if title: ax.set_title(title)
    return fig, ax

#--------------------------------------------------#
"""kmz forming functions"""

def make_kml(conv, figs, colorbar=None, **kw):
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw..."""
    
    llcrnrlon, llcrnrlat = conv[0].min(), conv[1].min()
    urcrnrlon, urcrnrlat = conv[0].max(), conv[1].max()
    kml = Kml()
    altitude = kw.pop('altitude', 2e3)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    kml.document.camera = camera
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='Legend')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.09, y=0.09,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    kml.savekmz(kmzfile)

def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=1024):
    """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image."""
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    if False:
        plt.ioff()  # Make `True` to prevent the KML components from poping-up.
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10)
    # KML friendly image.  If using basemap try: `fix_aspect=False`.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax
  
  
#------------------------------------------#
"""Print to csv functions"""

# Save as .csv file
def save_to_csv(ds, single, conv, pxSize, resolution, name):
    ds_arr = ds.shape
    ptslon = np.linspace(conv[0].min(),conv[0].max(),ds_arr[1]+1)[:-1]
    ptslon += (ptslon[1]-ptslon[0])/2
    ptslat = np.linspace(conv[1].min(),conv[1].max(),ds_arr[0]+1)[:-1]
    ptslat += (ptslat[1]-ptslat[0])/2
    lonm, latm = np.meshgrid(ptslon,ptslat)
    ds_q = int(np.floor(resolution/pxSize))
    arr = np.round(single[:-(single.shape[0] % ds_q),:-(single.shape[1] % ds_q)]).astype(int)
    most=np.zeros([len(range(0,arr.shape[0]-ds_q,ds_q))+1, len(range(0,arr.shape[1]-ds_q,ds_q))+1])
    for i in range(0,arr.shape[0],ds_q):
        for j in range(0,arr.shape[1],ds_q):
            most[int(i/ds_q),int(j/ds_q)] = Counter(arr[i:i+ds_q,j:j+ds_q].reshape(1,-1).tolist()[0]).most_common(1)[0][0]

    tdf = pd.DataFrame(np.concatenate([np.flip(latm.reshape(-1,1)), lonm.reshape(-1,1), ds.reshape(-1,1), most.reshape(-1,1)], axis=1), columns=['Latitude', 'Longitude', str(resolution)+'m T(C)', str(round(pxSize, 2))+'cm T(C)'])
    filt = tdf[ds.mask.reshape(-1,1) == False]
    filt.to_csv(name+'.csv', index_label='UID')