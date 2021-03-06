
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/original-undistorted.jpg "Undistorted"
[image2]: ./examples/original-undistorted2.jpg "Road Transformed"
[image3]: ./test_images/test1.jpg "Original Image"
[image4]: ./examples/undistorted-thresholded1.jpg "Undistorted Thresholded"
[image5]: ./examples/src-overlay-undistorted1.jpg "Src poly"
[image6]: ./examples/dst-overlay-warped1.jpg "Dst poly"
[image7]: ./examples/warped-lines1.jpg "Warp Example"
[image8]: ./examples/tracked1.jpg "Warp Example"
[image9]: ./examples/lines1.jpg "Fit Visual"
[image10]: ./examples/overlay-lines1.jpg "Output"
[video1]: ./output1_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README
All the code related to this project is
 [here](https://github.com/gvp-study/CarND-Advanced-Lane-Lines.git) . This contains the python files and the ipynb notebook I used to complete the project.  
The four main python files are:
* calibrate-camera.py: This file when called using 'python calibrate-camera.py' will read in the calibration images from the camera_cal directory, calibrate the camera and write the calibration parameters into a pickle file calibration_pickle.p.
* tracker.py: This file contains the main function called find_window_centroids that tracks the lane line in an image. The tracker class also keeps a history of the tracked lines if working with a video.
* find-lanes.py: This script is called by running 'python find-lanes.py'. This is the main python file that tracks the left and right lane lines in individual images and outputs the result on an overlaid image. The script also outputs the intermediate images during their processing from the raw image to a final image. The final image with the left and right lines overlaid on the original image is annotated with the curvature of the lane and the offset of the car from the center of the lane.
* video-gen.py: This script is called by 'python video-gen.py'. This python script functions exactly like find-lanes.py but works on reading images from a video and saves the resulting lane marking onto an output video called output1_video.mp4. Unlike find-lanes.py, video-gen.py uses a 15 long smoothing history to smooth over the line centers found in previous frames.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for calibrating a camera from a series of chessboard images is contained in the second code cell of the IPython notebook located in "./Advanced-Lane-Finder.ipynb" and the calibrate-camera.py file. The code when run generates the camera calibration parameters and saves them in the file ./calibration_pickle.p.  

I start by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for every calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the arrays `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The two main parameters that define the calibration are the camera projection matrix and the five distortion coefficients

| Projection Matrix       | Coefficients   |
|:-------------:|:-------------:|
| [[  1.15396093e+03   0.00000000e+00   6.69705357e+02] [  0.00000000e+00   1.14802496e+03   3.85656234e+02][  0.00000000e+00   0.00000000e+00   1.00000000e+00]] | [[ -2.41017956e-01  -5.30721173e-02  -1.15810355e-03  -1.28318856e-04  2.67125290e-02]] |

I applied this distortion correction to one of the calibration images using the `cv2.undistort()` function and obtained this result as shown below:

![alt text][image1]

I also stored the camera projection matrix and the distortion coefficients in a pickle file for use later.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the distortion correction function to one of the test images of the road and the distorted original and the undistorted road image is shown below:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholding functions to generate a binary image (thresholding steps at lines 163 through 167 in `find-lanes.py`). I tried both the gray image and the V-Channel in the HSV image to extract the edges. I found that the V-Channel was more robust in the presense of shadows and lane line color. All pixels in the V-Channel of the image where the both the gradient in x and y are above a threshold are considered to be a lane marker. The color threshold uses both the HLS and HSV coordinates and considers all pixels where the S and V values are above a threshold. This is to allow for more robustness in the line finding.

This is the input to the combined color + gradient thresholding pipeline.

![alt text][image3]

The output of this thresholding step is shown below.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The main thing for figuring out the mapping between the source and destination pixels using the warpPerspective function is to identify the images where the road lines are straight. This lets us visually check for the correctness of our src and dst points. I used the staight-lanes1.jpg images in the test_images directory.

Once the four points in the two images are identified, M = cv2.getPerspectiveTransform(src,dst) is used to get the forward perspective transform M. The inverse perspective transform Minv is also computed at the same time using the reversing the arguments Minv = cv2.getPerspectiveTransform(dst,src).

I chose the hardcode the source and destination points in the following manner based on the example.ipynb in the examples directory. I changed the fraction from 0.25 to 0.2 by visually looking at the output:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
frac = 0.2
dst = np.float32(
    [[(img_size[0] * (0.5-frac)), 0],
     [(img_size[0] * (0.5-frac)), img_size[1]],
     [(img_size[0] * (0.5+frac)), img_size[1]],
     [(img_size[0] * (0.5+frac)), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 192, 0        |
| 203, 720      | 192, 720      |
| 1127, 720     | 1088, 720      |
| 695, 460      | 1088, 0        |

The code to convert a given image to a warped view of the image from overhead is in lines 174 - 181 in `find-lanes.py`.  The cv2.warpPerspective() function takes as inputs the image and the forward transform M.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

![alt text][image6]
The warping transform can then used to transform any given image from the camera into a  view of the road from overhead. Another big advantage of the warping function is that it automatically segments out the region of interest that we chose. I used it to transform the thresholded image into a view of the lane lines as shown below.
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that I have the lane line pixels isolated, I used a window approach to progressively track the lane from the bottom of the image to the top. The call to the function find_window_centroids defined in tracker.py as shown below.
```python
window_centroids = curve_centers.find_window_centroids(warped)
```
The find_window_centroid function convolves a 25x80 pixel white rectangle on the 9 (80 pixel high) horizontal slices of the warped image using the np.convolve function. The centers of the left and right lines are found by looking at the peaks in the convolved output. The tracker starts at the bottom of the image and uses the line centers from the previous slice to restrict the search for the maximum in a tight window. When no line marking is found in a slice, it keeps the centroid from the previous slice.

The find_window_centroid also keep the line centers from all the previous smooth_factor number of frames to smooth out the results. Note that I set this factor to 1 in the find_lanes.py because the input are disparate frames used only for testing. I set it back to the default 15 when used in the video-gen.py.

After the window_centroids are computed in the image, I use the code in lines 247-261 in find-lanes.py to fit a 2nd order polynomial using the following code. Here I use the np.polyfit() function to find the lane line x value by feeding it the yaxis axis range from 0 to the image height. The resulting coefficients in left_fix and right_fix are then used to convert the yvals into left_fitx and right_fitx arrays for display.
```python

yvals = range(0,warped.shape[0])

res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

left_fit = np.polyfit(res_yvals,leftx,2)
left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
left_fitx = np.array(left_fitx,np.int32)

right_fit = np.polyfit(res_yvals,rightx,2)
right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
right_fitx = np.array(right_fitx,np.int32)

```

The result of the tracker is shown in the figure below.

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The polynomial coefficients can be used to analytically compute the radius of curvature of the left line using the formula as shown below.
* If the function is defined by f(y)=Ay**+By+C
* Curvature can be computed as R = ((1+(2Ay+B)^2)^1.5) / abs(2*A)

I found that the left and right lines differ in radius. I use only the left line curvature for display.
The offset of the center of the car from the middle of the lane is computed as the distance from the average of the left and right lines.
Note that the recalculation multiplies both coordinates by their corresponding meters_per_pixel factors so the resulting curvature and center offset are in meters.

```python
# Compute radius and center offset and convert from pixels to meters
ym_per_pix = curve_centers.ym_per_pix
xm_per_pix = curve_centers.xm_per_pix

curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,
                         np.array(leftx,np.float32)*xm_per_pix, 2)
curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix +
                 curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])

camera_center = (left_fitx[-1] + right_fitx[-1])/2
center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix

```
The result is two smooth curves that fit the left and right lines of the lane shown in blue and red in the figure below. Note that these are plotted in the warped overhead image view.

![alt text][image9]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The main step in this process is to convert all the drawings done in the warped top down view back to the normal image coordinates using the warpPerspective() function with the inverse projection transform Minv.
I implemented this step in the lines 283 through 284 in `find-lanes.py`   Here is an example of my result on a test image: Note that the radius of curvature and the offset from the center of the lane are also displayed.

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to the video output ./output1_tracked.mp4 that resulted from processing project_video.mp4.

![alt text][video1]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the main drawbacks to the approach used here is the sensitivity of the lane line thresholding due to brightness and shadows. This will become even more of a problem when the road conditions change depending on the time of day and the weather conditions such as rain. One way to fix this would be to try adaptive thresholding approaches based on the overall appearance of the image based on the intensity histogram.
This drawback of the approach is very apparent when I tested the video-gen.py on the challenge_video.mp4.

Another possible drawback could be the assumption that the transformation of the camera that is computed from the initial straight-lane.jpg remains the same through the rest of the test images and video. Any slope or tilt of the road or car could affect the assumption.

### References
Most of the code for the tracking and video making was written after looking at the video in the Project FAQ found here and the course notes. https://www.youtube.com/watch?v=vWY8YUayf9Q&feature=youtu.be
