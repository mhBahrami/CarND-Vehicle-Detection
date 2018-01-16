
# coding: utf-8

# # Udacity Self-Driving Car Engineer Nanodegree Program
# ## Vehicle Detection Project
# The goals / steps of this project are the following:
# 
# - Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# - Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# - Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# - Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# - Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# - Estimate a bounding box for vehicles detected.
# ---

# ### Import Packages

# In[1]:


from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pandas as pd
import pickle
import cv2
import glob
import time
import random

get_ipython().run_line_magic('matplotlib', 'inline')

print('>> Done!')


# ### Helper Functions
# The following code cell includes some helper functions which has been used in the rest of implementation.

# In[2]:


def plot_sample_data(files, titles, v_plot_count=1, fig_size=(8, 8), _axis = 'off'):
    h_plot_count = len(files)//v_plot_count
    fig, axs = plt.subplots(v_plot_count, h_plot_count, figsize=fig_size)
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for i in np.arange(len(files)):
        img = read_img(files[i])
        axs[i].axis(_axis)
        axs[i].set_title(titles[i], fontsize=10)
        axs[i].imshow(img)
        

def plot_sample_data_img(images, titles, v_plot_count=1, fig_size=(8, 8), _axis = 'off'):
    h_plot_count = len(images)//v_plot_count
    fig, axs = plt.subplots(v_plot_count, h_plot_count, figsize=fig_size)
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for i, img in enumerate(images):
        axs[i].axis(_axis)
        axs[i].set_title(titles[i], fontsize=10)
        axs[i].imshow(img)


def read_img(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# ### Training Data Set
# All images are the same shape `(64x64x3)` and in `*.png` format.

# In[3]:


car_images = glob.glob('./data_set/vehicles/**/*.png')
noncar_images = glob.glob('./data_set/non-vehicles/**/*.png')

car_images = shuffle(car_images)
noncar_images = shuffle(noncar_images)

print('>> Number of "cars" images: ',len(car_images))
print('>> Number of "non-cars" images: ',len(noncar_images))


# ### Plot Some Random Data

# In[4]:


n = 3
count = n*n
# car
sample_cars = shuffle(car_images, n_samples=count)
sample_cars_title = ['car']*count
# non-acar
sample_noncars = shuffle(noncar_images, n_samples=count)
sample_noncars_title = ['non-car']*count
# all
samples =np.concatenate((sample_cars, sample_noncars))
samples_title =np.concatenate((sample_cars_title, sample_noncars_title))
# plot
plot_sample_data(samples, samples_title, v_plot_count=n, fig_size=(count*1.5, n*1.5,))


# ### Convert Image to Histogram of Oriented Gradients (HOG)

# In[5]:


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    """
    A function to return HOG features and visualization
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# #### Calculate and Visualize HOG for One Sample Data

# In[6]:


# car sample
sample_car = shuffle(car_images, n_samples=1)[0]
img_car = read_img(sample_car)
_, img_car_hog = get_hog_features(img_car[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
sample_cars_title = ['car', 'car (HOG)']
# non-car sample
sample_noncar = shuffle(noncar_images, n_samples=2)[0]
img_noncar = read_img(sample_noncar)
_, img_noncar_hog = get_hog_features(img_noncar[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
sample_noncars_title = ['non-car', 'non-car (HOG)']
# all
samples= [img_car, img_car_hog, img_noncar, img_noncar_hog]
titles = np.concatenate((sample_cars_title, sample_noncars_title))
#plot
plot_sample_data_img(samples, titles, v_plot_count=2, fig_size=(6, 6))


# ### Extract HOG Features from an Array of Car and Non-Car Images

# In[7]:


def bin_spatial(img, size=(32, 32)):
    """
    A function to compute binned color features
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    - Define a function to compute color histogram features 
    - NEED TO CHANGE bins_range if reading ".png" files with mpimg!
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
    - A function to extract features from a list of images
    - Have this function call bin_spatial() and color_hist()
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
#         file_features = []
        # Read in each one by one
        image = read_img(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

#         if spatial_feat == True:
#             spatial_features = bin_spatial(feature_image, size=spatial_size)
#             file_features.append(spatial_features)
#         if hist_feat == True:
#             # Apply color_hist()
#             hist_features = color_hist(feature_image, nbins=hist_bins)
#             file_features.append(hist_features)
#         if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
#         file_features.append(hog_features)
#         features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# ### Feature Extraction
# Steps are as following:
# - Extract the features for data set.
# - Combine them
# - Define the labels
# - Shuffle and split
# 
# First, define some usefule parameters:

# In[8]:


colorspaces = {0:'YUV', 1:'RGB', 2:'HSV', 3:'LUV', 4:'HLS', 5:'YCrCb'}
hog_channels = {0:0, 1:1, 2:2, 3:'ALL'}
orients = [9, 11]

color_spase = colorspaces[0]
hog_channel = hog_channels[3]
orient = orients[1]
pix_per_cell = 8
cell_per_block = 2
split_ratio = 0.2


# Extract features from data set for:
# - 6 color spaces which is `colorspaces`
# - 1 HOG channel which is `hog_channel='All'`
# - 2 orient values which is `orient`
# - `pix_per_cell = 8`, `cell_per_block = 2`
# 
# in total 14 different combinations for extracting features.
# > **The folloing code cell takes ~8min to run.**

# In[ ]:


features = {'car': [], 'noncar':[], 'orient':[], 'cspace':[]}
for key in colorspaces.keys():
    color_spase = colorspaces[key]
    for orient in orients:
        # Extract the featurs for data set
        print('>> Extracting features for: color_spase=\'{0:5s}\', orient={1:2d}'.format(color_spase, orient))
        carf = extract_features(car_images, color_space=color_spase, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
        features['car'].append(carf)
        noncarf = extract_features(noncar_images, color_space=color_spase, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel)
    
        features['noncar'].append(noncarf)
        features['orient'].append(orient)
        features['cspace'].append(color_spase)
    
print('>> Done!')


# Build the training and testing data set. Training data set contains 80% of all data set and testing data contains 20% of it.

# In[12]:


def shuffle_and_split(car_features, noncar_features, ratio =0.2):
    ## Fit a per-column scaler - this will be necessary if combining 
    ## different types of features (HOG + color_hist/bin_spatial)
    ## Combine them
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    ## Fit a per-column scaler
    #X_scaler = StandardScaler().fit(X)
    ## Apply the scaler to X
    #scaled_X = X_scaler.transform(X)

    # Define the labels
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio,
                                                        random_state=rand_state)

    print('>> Training data set: features={0}, labels={1}'.format(len(X_train), len(y_train)))
    print('>> Testing data set: features={0}, labels={1}'.format(len(X_test), len(y_test)))
    
    return X_train, X_test, y_train, y_test


# ### Train Classifier
# I used `LinearSVC()`  as my classifier.

# In[13]:


def train_classifier(clf, X_train, X_test, y_train, y_test):
    # Check the training time for the SVC
    t=time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    ttrain = round(t2-t, 3)
    print('[+] {0} seconds to train SVC...'.format(ttrain))
    trained_classifier['ttrain'].append(ttrain)
    print('----------------------------------------------')
    
    count = 10
    accuracy = round(clf.score(X_test, y_test), 4)
    print('>> Accuracy = {0:7.4f}'.format(accuracy))
    
    # Check the prediction time for a single sample
    t=time.time()
    print('>> Predicted      : {0}'.format(clf.predict(X_test[0:count])))
    print('>> Expected labels: {0}'.format(y_test[0:count]))
    t2 = time.time()
    tpred = round(t2-t, 5)
    trained_classifier['tpred'].append(tpred)
    print('[+] {0} seconds to predict {1} labels with SVC.'.format(tpred, count))
    print('______________________________________________')
    
    return clf, accuracy


# In[14]:


trained_classifier = {'clf':[], 'acc':[], 'tpred':[], 'ttrain':[]}

for idx in range(len(features['cspace'])):
    print('[{0:2d}] color_spase={1:5s}, orient={2:2d}'.format(idx, features['cspace'][idx], features['orient'][idx]))
    car_features, noncar_features = features['car'][idx], features['noncar'][idx]
    X_train, X_test, y_train, y_test =         shuffle_and_split(car_features, noncar_features, ratio =0.2)
    lsvc = LinearSVC()
    lsvc, accuracy = train_classifier(lsvc, X_train, X_test, y_train, y_test)
    trained_classifier['clf'].append(lsvc)
    trained_classifier['acc'].append(accuracy)
    
    print(' ')


# ### Selecting the Classifirer
# First I sort the results based on the obtained accuracy:

# In[27]:


df_trained_classifier = pd.DataFrame(trained_classifier)
# print(df_trained_classifier)

df_features = pd.DataFrame(features)
# print(df_features)

data_frame = df_features.join(df_trained_classifier)
data_frame = data_frame.sort_values(by=['acc', 'tpred', 'ttrain'], ascending=False)
print(data_frame.filter(items=['cspace', 'orient', 'tpred', 'ttrain', 'acc']))


# Based on the above results, I choose the first one (`model_index = 6`) with higher accuracy `0.9952`.

# In[28]:


selected = 6
clf = trained_classifier['clf'][selected]
#X_scaler = trained_classifier['scaler'][selected]
color_space = features['cspace'][selected] # which is 'YUV'
orient = features['orient'][selected] # which is 9
hog_channel = hog_channels[3] # which is 'All'
pix_per_cell = 16
cell_per_block = 2


# ### Detecting Cars in a Frame Using Classifier
# `detect_cars()` is responsible to use every calculated information above to detect cars in a frame. **For implementation I used the code in the course.**

# In[35]:


test_images = np.array(glob.glob('./test_images/*.jpg'))
print(test_images)


# In[75]:


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(image)  
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
             # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

#             # Scale features and make a prediction
#             print(spatial_features.shape)
#             print(hist_features.shape)
#             print(hog_features.shape)
#             _t1=np.hstack((spatial_features, hist_features, hog_features))
#             _t2=_t1.reshape(1, -1)
#             print(_t2)
#             test_features = X_scaler.transform(_t2)    
#             #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
#             test_features = X_scaler.transform(np.hstack((spatial_features, hist_features)).reshape(1, -1))    
            
            test_prediction = svc.predict(hog_features.reshape(1, -1))
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img


# In[76]:


ystart = 400
ystop = 656
scale = 1.5
spatial_size = (32, 32)
hist_bins=32

idx = 0
img = read_img(test_images[idx])

out_img = find_cars(img, ystart, ystop, scale, 
                    clf, X_scaler, orient, 
                    pix_per_cell, cell_per_block, 
                    spatial_size, hist_bins)

plt.imshow(out_img)

