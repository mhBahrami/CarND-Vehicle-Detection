# Vehicle Detection and Tracking

[//]: #	"Image References"
[image1]: ./res/car_samples.png
[image2]: ./res/noncar_samples.png
[image3]: ./res/sample_hog_car.png
[image4]: ./res/sample_hog_noncar.png
[image5]: ./res/sliding_wins.png
[image6]: ./res/sample_sliding_wins.png
[image7]: ./res/sample_final_result.png

The main goal of the project is to create a software pipeline to identify vehicles in a video from a front-facing camera on a car.

Code:
- All commented code can be found at [`project-crowdai.ipynb`](https://github.com/mhBahrami/CarND-Vehicle-Detection/blob/master/project-crowdai.ipynb) jupyter notebook.
- `test_images/` contains images of road to test and fine-tune pipeline
- `output` includes the final generated video.

### Overview

Project code consist of following steps:

1. Load datasets
2. Extract features from datasets images
3. Train classifier to detect cars. (I used simple default `LinearSVM()`)
4. Scan video frame with sliding windows and detect hot boxes
5. Use hot boxes to estimate cars positions and sizes
6. Use hot boxes from previous steps to remove false positives hot boxes and make detection more robust

### Datasets

In this project I use two datasets. First is project dataset. It is splitted into [cars images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-car images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Here is examples of dataset images:

![alt text][image1]

![alt text][image2]

After playing with original dataset I found that it's not good enough in car detection and there are some false positive in the final result as well. “CrowdAI” dataset ([Annotated Driving Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations#annotated-driving-dataset), [Dataset 1](https://github.com/udacity/self-driving-car/tree/master/annotations#dataset-1)) solved this problem for me along with increasing performance of the classifier. I augment original dataset with 35725 car images and 41400 non-car images from “CrowdAI”. By changing proportion of original and  “crowdai” dataset images in training samples you may fine tune classifier performance.

> **Description from the repository:**
> he dataset includes driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across 9,423 frames collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans.

### Histogram of Oriented Gradients (HOG)

After playing with picture pixels, histogram and HOG features I decided to use only little amount of HOG features. My feature vector consist of 128 components which I extract from the `LUV` color space images. I believe that it is better to detect cars by only structure information and avoid color information because cars may have big variety of coloring. Small amount of features help to make the classifier faster while loosing a little amount of accuracy. My parameters of feature extraction are

```python
# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
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

Here is examples of my HOG features:

|     Car Sample      |   Non-Car Sample    |
| :-----------------: | :-----------------: |
| ![alt text][image3] | ![alt text][image4] |

### `LinearSVM` classifier

I decided to use `LinearSVM()` classifier and default `sklearn` parameters. With my small amount of features it shows time performance about 4 frames per second for whole pipeline. After combining datasets I use 17584 cars and 17936 non-cars images for training. Resulting accuracy is about 98.97%. Also I used [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) for feature normalization along resulting dataset. I found that it is really important step. It adds about 4% accuracy for my classifier.

### Sliding windows

For searching cars in an input image I use sliding window technics. It means that I iterate over image area that could contain cars with approximately car sized box and try to classify whether box contain car or not. As cars may be of different sizes due to distance from a camera we need a several amount of box sizes for near and far cars. I use 5 square sliding window sizes of 128, 96, 64, 48, and 32 pixels side size. While iterating I use 65%~75% window overlapping in horizontal and vertical directions. Here is an examples of sliding windows lattices which I use. 

![alt text][image5]

For computational economy and additional robustness areas of sliding windows don't convert whole image but places where cars appearance is more probable.

#### Test images

Ultimately I searched on two scales using `YCrCb` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

|   Sliding Windows   |
| :-----------------: |
| ![alt text][image6] |

|    Final Result     |
| :-----------------: |
| ![alt text][image7] |

Also in some frames, there are some **false positives** as well.

### Video processing

Same average boxes algorithm may be used to estimate cars base on last several frames of the video. We just need to accumulate hot boxes over number of last frames and then apply same algorithm here with higher threshold. 

Here's a [link to the video result](https://github.com/mhBahrami/CarND-Vehicle-Detection/blob/master/output/project_video_YCrCb.mp4) and you can watch it online [here](https://youtu.be/14IS37hfXpo).

### Conclusion

Detecting cars with SVM in sliding windows is interesting method but it has a number of disadvantages. While trying to make my classifier more quick I faced with problem that it triggers not only on cars but on other parts of an image that is far from car look like. So it doesn't generalizes well and produces lot of false positives in some situations. To struggle this I used bigger amount of non-car images for SVM training. Also sliding windows slows computation as it requires many classifier tries per image. Again for computational reduction not whole area of input image is scanned. So when road has another placement in the image like in strong curved turns or camera movements sliding windows may fail to detect cars.

I think this is interesting approach for starting in this field. But it is not ready for production use. I think convolutional neural network approach may show more robustness and speed. As it could be easily accelerated via GPU. Also it may let to locate cars in just one try. For example we may ask CNN to calculate number of cars in the image. And by activated neurons locate positions of the cars. In that case SVM approach may help to generate additional samples for CNN training.

