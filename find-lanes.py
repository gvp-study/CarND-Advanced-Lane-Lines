import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh=(10,255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

# Edit this function to create your own pipeline.
def color_threshold(img, s_thresh=(170, 255), v_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    # Threshold S channel
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary==1) & (v_binary==1)] = 1
    return output
    
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

images = glob.glob('./test_images/test*.jpg')
for idx, fname in enumerate(images):
    # Read in the image
    img = cv2.imread(fname)
    print('Undistorting ', idx, ' ', fname)
    # Undistort it.
    uimg = cv2.undistort(img, mtx, dist, None, mtx)
    # Make a preprocess image.
    preprocessImage = np.zeros_like(uimg[:,:,0])
    gradx = abs_sobel_thresh(uimg, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(uimg, orient='y', thresh=(25,255))
    c_binary = color_threshold(uimg, s_thresh=(100,255), v_thresh=(50,255))
    preprocessImage[((gradx==1) & (grady==1) | (c_binary==1))] = 255

    # Mapping the four points that make the perspective projection.
    img_size = (img.shape[1], img.shape[0]) # 1280 x 720
    print('Image size is ', img_size)
    bot_width = 0.80 # Bottom trapezoid width
    mid_width = 0.15 # Middle trapezoid width
    height_pct = 0.62 # Trapezoid height
    bottom_trim = 0.935 # From bottom to avoid the car hood
#    isrc = np.int32([[600, 500],
#                     [750, 500],
#                     [1180, 680], 1280-100, 
#                     [100, 680]]) 100
    slt = [img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct]
    srt = [img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct]
    srb = [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim]
    slb = [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]
    src=np.float32([slt, srt, srb, slb])
    print(src)
    dlt = [offset, 0]
    drt = [img_size[0]-offset, 0]
    drb = [img_size[0]-offset, img_size[1]]
    dlb = [offset, img_size[1]]
    offset = img_size[0]*0.1 # 320
    dst = np.float32([dlt, drt, drb, dlb])
    print(dst)
    # Perform the transforms.
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)
    # Write it out.
    print('Gradient + Color Thresholding Warp Perspective', idx, ' ', fname)
    write_name = './test_images/lanes_test'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, warped)

    write_name = './test_images/preprocessed_test'+str(idx+1)+'.jpg'
    isrc = np.int32([slt, srt, srb, slb])
    idst = np.int32([dlt, drt, drb, dlb])
    cv2.polylines(img, [isrc], True, (0,255,0), 3)
    cv2.polylines(img, [idst], True, (255,0,0), 3)
    cv2.imwrite(write_name, img)



