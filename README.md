## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[camera1]: ./img/camera.png "camera.png"
[Pipeline_distortion_img]: ./img/Pipeline_distortion_img.png "Pipeline_distortion_img"
[result]: ./img/result.png "result.png"

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

--------------------
### 1. Camera Calibration
Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I identify the feature points in the picture, then i use these feature point to process the calivration step.

```python
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.figure()
        plt.imshow(img)

# Test undistortion on an image
img = cv2.imread('./camera_cal/calibration3.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('output_images/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( ".wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
```
![camera1][camera1]
Define function for apply distortion
```python
def correctDistortion(img):
    ''' correct Distortion '''
    return cv2.undistort(img, mtx, dist, None, mtx)
```
--------

### 2. Pipeline (test images)

#### Provide an example of a distortion-corrected image.
![Pipeline_distortion_img][Pipeline_distortion_img]

#### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.
I have try many ways to process imgs, the result i found using sxbinary_s, sybinary_g and some color threshold is the most stable result. I also apply Roi in binary map.

```python
def processImg(rbgImg, s_thresh=(200, 255), sx_thresh=(60, 100), log=False):
    img = np.copy(rbgImg)    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)    
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]       
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    sxbinary_l = cal_abs_sobel(l_channel, 'x', thresh=sx_thresh)
    sybinary_l = cal_abs_sobel(l_channel, 'y', thresh=sx_thresh)

    sxbinary_s = cal_abs_sobel(s_channel, 'x', thresh=(30,200))   
    sybinary_g = cal_abs_sobel(gray, 'y', thresh=(30,200))

    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    layer = np.zeros_like(s_binary)
    layer[(l_binary == 1) | (b_binary == 1) | (s_binary==1) | (sxbinary_s==1) | (sybinary_g == 1)] = 1

    color_binary = np.dstack(( layer, layer, layer))

    imshape = color_binary.shape

    color_binary = region_of_interest(color_binary,\
                                      np.array(\
                                               [[(40,imshape[0]),\
                                                 (imshape[1]/2-50, imshape[0]*3/5-40), \
                                                 (imshape[1]/2, imshape[0]*3/5-40), \
                                                 (imshape[1]-50,imshape[0])]], dtype=np.int32)
                                     )
    return color_binary
```

#### Perspective transform
Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
i use cv2.getPerspectiveTransform to transform picture from src rectangle to target rectangle.
```python
srcPt = np.float32([
    [585, 460],
    [203, 720],
    [1127, 720],
    [695, 460]
])

dstPt = np.float32([
    [320, 0],
    [320, 720],
    [960, 720],
    [960, 0]
])

M = cv2.getPerspectiveTransform(srcPt, dstPt)
Minv = cv2.getPerspectiveTransform(dstPt, srcPt)

def transformImgToBirdEyeView(img):
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

#### Line fitting
Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use a sliding window to get the valid points in the picture, then i use np.polyfit to get the polynomial.

```python
gray_img = out_img
out_img = np.copy(gray_img)
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

#### calculate curvature and offset
Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use the standard ratio to calculate the curverad, then using yAxis value to calculate the curverad.
```python

ym_per_pix = 15/720 # meters per pixel in y dimension
xm_per_pix = 3.7/900 # meters per pixel in x dimension

def calCurvature(yVals, x):
    y_eval = np.max(yVals)

    fit_cr = np.polyfit(ploty*ym_per_pix, x*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
#     print(curverad, 'm')
    return curverad
```
First i get maximum left and right point, then i compare the center of img with the center of left_right.   
```python
def findOffCenter(img, left_fitx, right_fitx):
    '''
        Find how far the car off the center
    '''
    position = img.shape[1]/2
    left  = left_fitx[-1] #np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = right_fitx[-1] #np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    res = (640 - center)*xm_per_pix
#     print(res, 'm')
    return res
```

#### Draw result
Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
![result][result]
```python
def drawOffCenter(rgbImg, position):
    rgbImg = rgbImg.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(rgbImg,text,(400,150), font, 1,(255,255,255),2)
    return rgbImg

def drawCurvature(rgbImg, curvature):    
    rgbImg = rgbImg.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(int(curvature))
    cv2.putText(rgbImg,text,(400,100), font, 1,(255,255,255),2)
    return rgbImg
```

---

### Pipeline (video)

    Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)
---
<a href="http://www.youtube.com/watch?feature=player_embedded&v=oOUIOLuphRQ" target="_blank"><img src="http://img.youtube.com/vi/oOUIOLuphRQ/0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

----
### discussion
 Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
1. compuration speed
  Hough transform need a heavy computation compare to canny edge method.
2. Hard time in tunning.
  Tuning the threshold is a hard time, i use most of the time in tuning those params.
##### Potential improvement
1. Use GPU it gives a pseed improvement
2. Use Neural network For image segmentation.
