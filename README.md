## Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[dataset]: ./output_images/dataset.png
[yuv]: ./output_images/yuv.png
[features]: ./output_images/features.png
[rects]: ./output_images/rects.png
[regions]: ./output_images/regions.png
[regions_rects]: ./output_images/regions_rects.png
[heat]: ./output_images/heat.png


[video1]: ./project_video.mp4

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Everything starts with data, so let us do same way.
First of all I've loaded provided dataset, then I've read all the file names into `non_vehicles_images_names` and
`vehicles_images_names` variables. This code could be found in the third cell of the [notebook](https://github.com/kav137/CarND-Vehicle-Detection/blob/master/pipeline.ipynb)

Here is a visualization of some randomly picked images from the dataset:

![Dataset visualization][dataset]

The code for this step could be found [features.py](https://github.com/kav137/CarND-Vehicle-Detection/blob/master/src/features.py)
file. The code itself is based on the implementation provided within lessons. I've refactored it a bit, but in general
the conecpt remained the same:

In order to extract HOG features from the image we have to convert to the appropriate colorspace and apply skimage's
HOG method for single channel (if any selected) or for each channel concating the result afterwards.

Two methods that are defined within the module are:

* [get_hog_featres](https://github.com/kav137/CarND-Vehicle-Detection/blob/795d214ba3731b52ddcb62f6b7b69934d1a08cbe/src/features.py#L8)
* [extract_features](https://github.com/kav137/CarND-Vehicle-Detection/blob/795d214ba3731b52ddcb62f6b7b69934d1a08cbe/src/features.py#L21)

Both of those methods have default parameters values (I've set them to the required ones), but they could be changed
when method is called.

#### 2. Explain how you settled on your final choice of HOG parameters.

I've chosen YUV colorspace as a target colorspace to be used for HOG feature extraction. And there were some reasons for it :)
After I've defined the code which allow to extract features from the image, I've performed some investigation about the
performance and accuracy for different colorspaces.
The code I've used for it could be found in [experiments.ipynb](https://github.com/kav137/CarND-Vehicle-Detection/blob/master/experiments.ipynb)

Here are the results I've got:

Colorspace | Orientations | Pixels Per Cell | Cells Per Block | Evaluation time | Accuracy
--- | --- | --- | --- | --- | ---
RGB | 9 | 8 | 2 | 11.07 | 0.9758
HLS | 9 | 8 | 2 | 5.63 | 0.9814
HSV | 9 | 8 | 2 | 6.33 | 0.9828
YUV | 11 | 8 | 2 | 6.03 | 0.9825
LUV | 9 | 8 | 2 | 6.61 | 0.9764
YCrCb | 9 | 8 | 2 | 5.46 | 0.9797
YCrCb | 11 | 8 | 2 | 5.93 | 0.9823
YUV | 11 | 16 | 2 | 0.83 | 0.9828

As you can see as a parameter YUV colorspace leads us to the most accurate result.

After I've selected the colorspace to use, I've played with some other parameters and here are the final ones:

Parameter name | value
--- | ---
orient | 11
pix_per_cell | 16
cell_per_block | 2
hog_channel | 'ALL'
colorspace |'YUV'

Below you can see the visualization of parametrs that were chosen. The first three rows representing vehicles.
Images from left to right:

1. original image
2. YUV image
3. Apllied HOG (Y channel)
4. Apllied HOG (U channel)
5. Apllied HOG (V channel)

![yuv][yuv]

You can see how HOG values differs itself - it is great because basing on this values classifier will be able to
make the right choice :)

Here you can see how values are distributed across histogram of HOG:

![features]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As for learning tool SVM classifier have been chosen. Why? Because I want to know which way is better for this task:
SVM or neural network (with which I'm already familar). And this project is a nice place to give a SVM a try.
Later I want to do this project using CNNs and see which approach is better.

I trained a linear SVM with the default classifier parameters and using HOG features alone. Spatial features or histograms
were not so useful as HOG, so I've decided to use only HOG. The accuracy I've achieved is: 98.28%.

This code is contained in sixth cell in [notebook](https://github.com/kav137/CarND-Vehicle-Detection/blob/master/pipeline.ipynb)
Look for title "Train classifier" (it is much easier than count cells:)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

As a general concept I've used find_images function from the lesson.
All the code could be found in [search.py](https://github.com/kav137/CarND-Vehicle-Detection/blob/795d214ba3731b52ddcb62f6b7b69934d1a08cbe/src/search.py#L21) module.

The function itselfs accepts image, _y_ coordinates which are treated like ROI and all the rest parameters we've discussed earlier.
As an output we've got an array of rectangles which represent found cars.

Here is the result:

![rects][rects]

The dimensions and proportions of the car can vary, so both the size of the window and the search area should change too. I add function find_cars_with_search_regions, which allows to reduce the number of windows and to depend less on the scale.

The main principle is: the closer (lower) the car is - the larger it would appear, so we should set window sizes accordingly

Example of using different params of searching regions:

![regions][regions]

The code for this function could be also found in [search.py](https://github.com/kav137/CarND-Vehicle-Detection/blob/795d214ba3731b52ddcb62f6b7b69934d1a08cbe/src/search.py#L80)

The values used as search regions are:

Y (top) | Y (bottom) | Scale
--- | --- | ---
400 | 475 | 1.0
410 | 525 | 1.0
405 | 525 | 1.25
405 | 550 | 1.5
410 | 550 | 1.5
410 | 585 | 1.75
415 | 585 | 2
425 | 700 | 2.85

Here is an example of image processed using search regions:

![regions_rects][regions_rects]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline for image processing is seemed to be simple (but only as a concept - implementation is not) :)

1. Take an image and convert it to the appropriate colorspace
2. Extract features using sliding window techinque and HOG extractor
3. Feed the values to the classifier
4. Collect all the positive results
5. Filter false positives

Code with pipeline could be found in notebook under the title "Define a pipeline for video processing"
None optimizations besides regions were provided

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a (github) [link to my video result](./project_processed_video.mp4)
Here's a (google drive)[link to my video result](https://drive.google.com/file/d/1cHCy7UpBnZPrtV7aGSctscfHEZXHNFoj/view?usp=sharing)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to avoid overlapping and some false positives I've used heatbox technique. The gist of this approach is that
we collect all the rects we've found and after that we analyze to which area multiple rects are pointing and for
which there are just single. After we khow it we can create a heatmap: the more rects - the higher the temperature.
Based on this we can filter some noizy positives and create a rough selection for the areas which are "hot".

Individual blobs were identified using `scipy.ndimage.measurements.label` method.

Here you can see how this approach works:

![heat][heat]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are multiple problems that such approach faces:
1. Performance. I have a powerful enough PC, but all the processing evaluates in time comparable to the video's length.
I guess in case such system have to work in real time it is likely to fail because some compoutations are heavy and
could provide delays. of course the algorithm should be optimized
2. Image processing. The filter selected works fine for provided whether and lightning conditions, but in case they
would be different pipeline may fail
3. Algorithm do not recognizes cars which are far enough - on the high speed it could be crucial.
