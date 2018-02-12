# Vehicle Detection and Tracking

[//]: #	"Image References"
[image1]: ./res/car_samples.png
[image2]: ./res/noncar_samples.png
[image3]: ./res/sample_hog_car.png
[image4]: ./res/sample_hog_noncar.png
[image5]: ./res/sliding_wins.png
[image6]: ./res/sample_sliding_wins.png
[image7]: ./res/sample_heatmap.png
[image8]: ./res/sample_heatmap_th.png
[image9]: ./res/sample_heatmap_th_label.png
[image10]: ./res/sample_refine.png
[image12]: ./res/sample_final_result.png

## The Goal of this Project

In this project, your goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. The test images and project video are available in [the project repository](https://github.com/udacity/CarND-Vehicle-Detection).

This project contains:

- The implementation can be found at [`project-final.ipynb`](https://github.com/mhBahrami/CarND-Vehicle-Detection/blob/master/project-final.ipynb) jupyter notebook.


- `test_images/` contains images of the road to test and fine-tune the pipeline.
- `output` includes the final generated video.
- `res` is a folder which includes the images of this *“README.md”*.

### Overview

Main steps of implementing the car detection:

1. Load datasets (Add more data as needed to make better dataset)
2. Extract features from datasets images (here the feature is the HOG features)
3. Train classifier to detect cars (I used `LinearSVM()`)
4. Scan video frames with sliding windows and detect the hot boxes
5. Use the hot boxes for the first estimation of the cars position
   - Use threshold to remove some false positive results
6. Use this estimation for detecting the cars positions more accurately
7. Save hot boxes from the previous step for the last 10 frames and use them to remove false positives hot boxes and make detection more robust
   - To do that I made an average over the last 9 frames plus current frame

I would explain more later.

### Datasets

In this project I used two datasets. First is project dataset. It is splitted into [cars images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-car images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Here is examples of dataset images:

![alt text][image1]

![alt text][image2]

After playing with original dataset I found that it's not good enough in car detection and there are some false positives in the final result as well. First I used “CrowdAI” dataset ([Annotated Driving Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations#annotated-driving-dataset), [Dataset 1](https://github.com/udacity/self-driving-car/tree/master/annotations#dataset-1)) to solve this problem.

> **Description from the repository:**
> he dataset includes driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across 9,423 frames collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans.

Unfortunately, using this dataset results in decreasing the accuracy and consequently increasing the false positives in my model. So, I preferred to generate more dataset myself from the project video. And this time I gained more accurate detection.

### Feature extraction

After playing with the different combinations of HOG (One channel or `'ALL'` of them), Spatial, and Histogram features. I decided to use all of them. 

#### Histogram of Oriented Gradients (HOG)

After playing with picture pixels, histogram and HOG features I decided to use only little amount of HOG features. Only the first channel `hog_channel = 0`.

##### Examples of my HOG features

|                          Car Sample                          |                        Non-Car Sample                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![img](file://D:\Mohammad\Education\Udacity\carnd\CarND-Vehicle-Detection\res\sample_hog_car.png?lastModify=1518416242) | ![img](file://D:\Mohammad\Education\Udacity\carnd\CarND-Vehicle-Detection\res\sample_hog_noncar.png?lastModify=1518416242) |

#### Color Space

I tested multiple color spaces but only the results of `'YCrCb'` satisfied me. You can made a better detection in this color space.

#### Tuned Parameters

You can see the tuned parameters here:

```python
# Define parameters for feature extraction
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

### `LinearSVM` classifier

I decided to use `LinearSVM()` classifier and default `sklearn` parameters. After combining datasets I use 18634 cars and 17936 non-cars images for training. I used 80% of them for training and 20% for testing. Also I used `shuffle` of `sklearn.utils` to make better combinations of dataset for training and testing. The accuracy is about 97.99%. Also I used [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for feature normalization along resulting dataset. I found that it is really important step. It adds about 4% accuracy for my classifier. 

The summary of the trained classifier is as following:

```python
>> Car samples:  18634
>> Notcar samples:  17936
>> Using: 8 orientations 8 pixels per cell And 2 cells per block
>> Feature vector length: 2432
>> 11.85 Seconds to train SVC...
>> Test Accuracy of SVC =  0.9799
```

### Sliding windows

For searching cars in a frame I used sliding window technics. It means that I iterate over image area that could contain cars with approximately car sized box and try to classify whether box contain car or not. As cars may be of different sizes due to distance from a camera we need a several amount of box sizes for near and far cars. I use 5 square sliding window sizes of 128, 96, 64, 48, and 32 pixels side size. While iterating I use 65%~75% window overlapping in horizontal and vertical directions. Here is an examples of sliding windows lattices which I used to detect cars at the right side. 

| Sliding windows used for right side car detection |
| :-----------------------------------------------: |
|                ![alt text][image5]                |

For computational economy and additional robustness areas of sliding windows don't convert whole image but places where cars appearance is more probable.

#### Test images

Ultimately I searched for cars in my sample images which you can find the in the **test-images** folder. The results as follows:

##### Sliding Windows

##### ![alt text][image6]

##### Heatmap

This the function for generating the Heatmap image from the obtained rectangles from the previous step:

```python
def add_heat(heatmap, bbox_list, margin=10):
    img = np.copy(heatmap)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        area = (box[0][1]-box[1][1])*(box[0][0]-box[1][0])
        if (area < 1200): margin = 0
        img[box[0][1]-margin:box[1][1]+margin, box[0][0]-margin:box[1][0]+margin] += 1

    # Return updated heatmap
    return img
```

As you can see I'm filtering out the very small rectangles. For detecting these rectangles, I calculate the area of rectangle and if it is smaller than 1200 I will remove them. Also, I will increase the size of detected rectangles by a value called `margin` to generate a better Heatmap image. The default value for `margin` is 10.

The result is as follows:

![alt text][image7]

##### Heatmap with threshold

In this step I will filter out more false positive rectangles by applying a threshold to the Heatmap. The function for this step is as follows:

```python
def apply_threshold(heatmap, threshold):
    img = np.copy(heatmap)
    # Zero out pixels below the threshold
    img[heatmap <= threshold] = 0
    # Return thresholded map
    return img
```

I used a `threshold=3` for this step.

The result is as following:

![alt text][image8]

##### Apply SciPy Labels to Heatmap

In this step I did 2 things to generate first estimation for the cars position:

1. Using `label` function from `scipy.ndimage.measurements`.
2. Removing other false positive rectangles by calculating the area.

This is the function for this step:

```python
def get_labels(img_th):
    labels = label(img_th)
    _map = labels[0]
    _count = labels[1]
    for i in range(1,_count+1):
        while(1):
            _num = np.count_nonzero(_map==i)
            if(_num > 0 and _num < 1500):
                a = np.zeros_like(_map)
                a[_map>i] = 1
                _map[_map==i] = 0
                _map=_map-a
                _count-=1
            else:
                # Find pixels with each car_number label value
                nonzero = (_map==i).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                break
    return (_map, _count)
```

And the result is as following:

![alt text][image9]

##### Refining the detections

In this step I will use the first estimation to draw a better rectangle around the car. To accomplish it, I used the detected area in the previous step and added a margin to its sides. Then I used the same technique to detect car. After this the result is better but is slows down the processing of each frame a little bit.

> **NOTE**
>
> I save the result of this step in a buffer for smoothing the next Heatmaps images. It really helps to reduce false positive rectangles.

The result is as following:

![alt text][image10]

### `process_frame()` Function

The `process_frame()` function is the function which I used to process video frames.

```python
def process_frame(img, th=3, heatmap_smothing=True, fine_proc = True):
    global buffer
    img_bgr = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2BGR)
    windows = all_sliding_windows(img_bgr)
    
    hot_windows = search_windows(img_bgr, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)    
    
    heatmap_img1 = np.zeros_like(img_bgr[:,:,0])
    heatmap_img2 = add_heat(heatmap_img1, hot_windows)
    heatmap_img3 = smooth_heatmap(heatmap_img2, add_to_buffer=False) if heatmap_smothing else heatmap_img2
    heatmap_img4 = apply_threshold(heatmap_img3, th)
    labels1 = get_labels(heatmap_img4)
    img_draw, rects1 = draw_labeled_bboxes(np.copy(img), labels1, draw=True)                 
    if(fine_proc and len(rects1)>0):
        windows = get_fine_windows(img, rects1, img.shape[1], img.shape[0])
        img_draw = fine_process_frame(np.copy(img), img_bgr, windows, heatmap_smothing=heatmap_smothing)
    return img_draw
```

This is all the steps I described in the previous section. Here you can see a new function called `smooth_heatmap()` 

```python
def smooth_heatmap(heatmap, add_to_buffer=False):
    global buffer
    if(add_to_buffer):
        if(buffer is None):
            buffer = [heatmap]
        else:
            buffer += [heatmap]
        buffer = buffer[-10:] if len(buffer)>10 else buffer
        smth_hmap = np.int32(np.average(buffer, axis=0))
    else:
        if(buffer is None):
            smth_hmap = heatmap
        else:
            smth_hmap0 = buffer[-10:] if len(buffer)>10 else buffer
            smth_hmap0 += [heatmap]
            smth_hmap = np.int32(np.average(smth_hmap0, axis=0))
    return smth_hmap
```

which will use the obtained Heatmaps in the last 10 previous frames for generating a more accurate Heatmap for the current frame. It really **reduces the false positive rectangles** and results in a smooth change of the drawn rectangles around the car.

### Video processing

I used the `process_frame()` function to process the frames.

Here's a [link to the video result](https://github.com/mhBahrami/CarND-Vehicle-Detection/blob/master/output/project_video.mp4) and you can watch it online [here](https://youtu.be/_R0Wq6NPAOk).

### Conclusion

For car detection you can use SVC classifiers. However, using a `SVC` is not enough to detect cars in the video frames. To train this classifier we should very good dataset as well. Moreover, using a Scaler help the classifier to predict the results more accurately. Training the SVC and predicting the results is not enough for car detections. Even with a good dataset and a well-trained classifier there would be some false positive predictions. To eliminate them we should use other techniques like Heatmap, applying threshold, and average of previous frames' results. After doing all this work you can have a good prediction. 

##### Discussion

- Of course, the algorithm may fail in case of difficult light conditions, which could be partly resolved by the classifier improvement.
- It is possible to improve the classifier by additional data augmentation, hard negative mining, classifier parameters tuning etc.
- The algorithm may have some problems in case of car overlaps another. To resolve this problem one may introduce long term memory of car position and a kind of predictive algorithm which can predict where occluded car can be and where it is worth to look for it.
- To eliminate false positives on areas out of the road, one can deeply combine results from the Advanced Lane Line finding project to correctly determine the wide ROI on the whole frame by the road boundaries. 
- The pipeline is not a real-time. One can further optimize number of features and feature extraction parameters as well as number of analyzed windows to increase the rate because lane line detection is quite fast.

### License

[MIT License](LICENSE).