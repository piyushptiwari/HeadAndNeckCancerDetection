import csv
import mat73
import cv2
import pandas as pd
import numpy as np
import pydicom as dicom
import dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from xtract_features.helpers import *
from xtract_features.glcms import *
from xtract_features.extract import *

from xtract_features.twodconv import conv2d
import scipy.ndimage as ndi
from scipy.stats import skew
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')


data_path = "C:\\Users\\peace\\Desktop\\12\\HN-HGJ-001\\CT\\image\\"
masks_path='C:\\Users\\peace\\Desktop\\12\HN-HGJ-001\\CT\\masks\\'
overlayed_path='C:\\Users\\peace\\Desktop\\12\HN-HGJ-001\\CT\\overlayed\\'
jpg_path='C:\\Users\\peace\\Desktop\\12\HN-HGJ-001\\CT\\jpg\\'
masked_path='C:\\Users\\peace\\Desktop\\12\HN-HGJ-001\\CT\\masked\\'
rtss_path='C:\\Users\\peace\\Desktop\\12\\HN-HGJ-001\\CT\\RTS\\'
g = glob(data_path + '/*.dcm')
data_dict = mat73.loadmat(rtss_path+'RTSS.mat')
data = np.array(data_dict)
a=data_dict['contours']['Segmentation'].astype(np.uint8)
lst_numpy_arrs, ids = extract_img_array(data_path, getID=True)
print(a.shape)

print(ids[:10])

#Convert all dicom images to nparray and putting all arrays to npy file
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)
np.save("fullimages_%d.npy" % (id), imgs)
file_used="fullimages_%d.npy" % id
n_array = np.load(file_used).astype(np.float64)

#show every 3rd image from 10th image from the npy file
def sample_stack(stack, start_with=10,rows=6,cols=6, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(n_array)

plt.hist(n_array.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.xlim(-2000, 2000)
plt.ylabel("Frequency")
plt.show()

#one Example of masked image and its features that we are extracting
c=a[72]*n_array[72]
plt.imshow(a[72])
plt.show()

print("Mean intensity")
print(cv2.mean(c))
print("varience")
print(np.var(c))
print("skewness")
print(skew(c, axis=0, bias=True))
print("kurtosis")
print(kurtosis(c, axis=0, bias=True))
print("Shannon's entropy for image ")
print(s_entropy(c))
print("Simple entropy for image ")
print(entropy_simple(c))
feats = glcm(c)

# correlation
corr = feats.correlation()
print("correlation ")
print(corr)

# homogeneity
homogeneity = feats.homogeneity()
print("homogeneity ")
print(homogeneity)

# contrast
cont = feats.contrast()
print("contrast")
print(cont)

# energy
energy = feats.energy()
print("energy ")
print(energy)


'''
# all glcm features at once
allf = feats.glcm_all()
print("array masked image ")
print(allf)
'''

"""Image moments are used to describe objects after segmentation and play an
essential role in object recognition and shape analysis. Images moments may be
employed for pattern recognition in images. Simple image properties derived via
raw moments is area or sum of grey levels. _moments is a list of 24 variant
moments and _hu_moments is list of the 7 hu moments which are invariant."""

from xtract_features.moments import *

hu_moments = moments(c).get_HuMoments()
print("printing hu moments")
print(hu_moments)

moments = moments(c).get_moments()
print("printing moments")
print(moments)


from xtract_features.region_props import *
rp = region_props(c)
# maximum area region
max_area = rp.max_area()
print("printing max_area")
print(max_area)


# plot black and white
rp.plot_show_bw()
print("plot black and white")
plt.show()
# mean of areas of all the regions
rp.mean_area()
print("mean of areas of all the regions")
print(rp.mean_area())
# eccentricity of the highest area region
rp.eccentricity()
print("eccentricity of the highest area region")
print(rp.eccentricity())

rp.euler_number()
print("rp.euler_number()")
print(rp.euler_number())

rp.solidity()
print("rp.solidity()")
print(rp.solidity())

rp.perimeter()
print("rp.perimeter()")
print(rp.perimeter())

# standard deviation of all the areas of the regions of the given image
rp.std_area()
print("rp.std_area()")
print(rp.std_area())

# otsu’s Threshold
rp.thresh_img()
print("erp.thresh_image()")
print(rp.thresh_img())

rp.bb()
print("rp.bb()")
print(rp.bb())

rp.bb_area()
print("rp.bb_area()")
print(rp.bb_area())

rp.centroid_r()
print("rp.centroid_r()")
print(rp.centroid_r())

rp.convex_area_r()
print("rp.convex_area_r()")
print(rp.convex_area_r())

rp.coordinates_r()
print("rp.coordinates_r()")
print(rp.coordinates_r())

rp.eq_diameter()
print("rp.eq_diameter()")
print(rp.eq_diameter())

rp.extent_r()
print("rp.extent_r()")
print(rp.extent_r())

rp.filled_area_r()
print("rp.filled_area_r()")
print(rp.filled_area_r())

rp.inertia_tensor_r()
print("rp.inertia_tensor_area()")
print(rp.inertia_tensor_r())

rp.label_r()
print("erp.label_r()")
print(rp.label_r())

rp.inertia_tensor_eigvals_r()
print("rp.inertia_tensor_eigvals_r()")
print(rp.inertia_tensor_eigvals_r())

rp.local_centroid_r()
print("rp.local_centroid_r()")
print(rp.local_centroid_r())

rp.maj_ax_len()
print("rp.maj_ax_len()")
print(rp.maj_ax_len())

rp.min_ax_len()
print("rp.min_ax_len()")
print(rp.min_ax_len())

rp.orient()
print("rp.orient()")
print(rp.orient())

#demo
#14 kernal names:identity edge-all edge-H edge-V sharp gauss-3 gauss-5 boxblur unsharp
#gradient-H gradient-V sobel-H sobel-V emboss

z=conv2d(c, "identity")
print("identity")
print(z)

z=conv2d(c, "edge-all")
print("edge-all")
print(z)

z=conv2d(c, "edge-H")
print("edge-H")
print(z)

z=conv2d(c, "edge-V")
print("edge-V")
print(z)

z=conv2d(c, "sharp")
print("sharp")
print(z)

z=conv2d(c, "gauss-3")
print("gauss-3")
print(z)

z=conv2d(c, "gauss-5")
print("gauss-5")
print(z)

z=conv2d(c, "boxblur")
print("boxblur")
print(z)

z=conv2d(c, "unsharp")
print("unsharp")
print(z)

z=conv2d(c, "gradient-H")
print("gradient-H")
print(z)

z=conv2d(c, "gradient-V")
print("gradient-V")
print(z)

z=conv2d(c, "sobel-H")
print("sobel-H")
print(z)

z=conv2d(c, "sobel-V")
print("sobel-V")
print(z)

z=conv2d(c, "emboss")
print("emboss")
print(z)


for i in range(0,len(g)):
    j=a[i]
    if sum(sum (j))!=0:
        plt.imsave(masks_path+str(i)+'mask.jpg',a[i])
        plt.imsave(jpg_path+str(i)+'image.jpg',n_array[i])
        c=a[i]*n_array[i]
        plt.imsave(masked_path+str(i)+'masked.jpg',c)
        plt.imshow(c, interpolation='None', cmap='gray',alpha=0.5)
        #plt.show()
'''
#features for all dicom images
dataframe = get_df_from_img_array(n_array,ids, getId = True)
print("Extracting features for all images : ")
df = pd.DataFrame(dataframe)
df.to_csv('1.csv')
'''

'''
#from dcm images features without id
dfa = get_df_from_path(data_path , [])
print("5 df from top are: ")
df = pd.DataFrame(dfa)
df.to_csv('1.csv')
'''

#features for masked images
X_data = []
masked_files = glob(masked_path+'*.jpg')
for myFile in masked_files:
    masked_image = cv2.imread (myFile)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    X_data.append (gray_image)
#print('X_data shape:', np.array(X_data).shape)
dataframe = get_df_from_img_array(X_data,ids, getId = True)
df = pd.DataFrame(dataframe)
print("Extracted features for masked images : ")
df.to_csv('2.csv')


#features for overlayed images
masks_files = glob(masks_path+'*.jpg')
jpg_files = glob(jpg_path+'*.jpg')
for i in range(0,len(masked_files)):
    img=cv2.imread(masks_files[i])
    im2=cv2.imread(jpg_files[i])
    dst = cv2.addWeighted(img, 0.5, im2, 0.5, 0)
    plt.imsave(overlayed_path+str(i)+'overlayed.jpg',dst)
    
Y_data = []
overlayed_files=glob(overlayed_path+'*.jpg')
for myFile in overlayed_files:
    overlayed_image = cv2.imread (myFile)
    gray_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2GRAY)
    Y_data.append (gray_image)
#print('X_data shape:', np.array(X_data).shape)
dataframe = get_df_from_img_array(Y_data,ids, getId = True)
print("Extracted features for overlayed images : ")
df = pd.DataFrame(dataframe)
df.to_csv('3.csv')


        
