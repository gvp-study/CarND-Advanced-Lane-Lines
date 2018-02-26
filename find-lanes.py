import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tracker import tracker

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh_gray(img, orient='x', thresh=(10,255)):

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

def abs_sobel_thresh(img, orient='x', thresh=(10,255)):

    # Apply the following steps to img
    # 1) Convert to HLS
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobel = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1)
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

# Makes a window mask around pixels in the rows at the height level
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

# Makes a mask about a region of interest defined by the vertices polygon.
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill colo
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
dir = 'test_images'
#dir = 'test_images2'
images = glob.glob('./'+dir+'/test*.jpg')
#images = glob.glob('./'+dir+'/challenge*.jpg')

# Go over the list of images.
for idx, fname in enumerate(images):
    # Read in the image
    img = cv2.imread(fname)

    img_size = (img.shape[1], img.shape[0]) # 1280 x 720
    # Source and destination polygons to warp the undistorted image to overhead view.
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
    isrc = np.int32(src)
    idst = np.int32(dst)
    
    print('Undistorting ', idx, ' ', fname)
    # Undistort it.
    uimg = cv2.undistort(img, mtx, dist, None, mtx)
    # Write it.
    udimg = np.copy(uimg)
    write_name = './'+dir+'/undistorted'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, udimg)

    # Write it.
    soimg = np.copy(uimg)
    cv2.polylines(soimg, [isrc], True, (0,255,0), 3)
    cv2.polylines(soimg, [idst], True, (255,0,0), 3)
    write_name = './'+dir+'/src-overlay-undistorted'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, soimg)

    # Make a preprocess image.
    preprocessImage = np.zeros_like(uimg[:,:,0])
#    gradx = abs_sobel_thresh_gray(uimg, orient='x', thresh=(20,100))
#    grady = abs_sobel_thresh_gray(uimg, orient='y', thresh=(20,100))
#    c_binary = color_threshold(uimg, s_thresh=(100,255), v_thresh=(50,255))
#    preprocessImage[((gradx==1) & (grady==1) | (c_binary==1))] = 255

    gradx = abs_sobel_thresh(uimg, orient='x', thresh=(20,100))
    grady = abs_sobel_thresh(uimg, orient='y', thresh=(20,100))
    c_binary = color_threshold(uimg, s_thresh=(100,255), v_thresh=(50,255))
    preprocessImage[((gradx==1) & (grady==1) | (c_binary==1))] = 255

    # Mapping the four points that make the perspective projection.
    # This is the warped overhead look.
    print('Image size is ', img_size)
    print(isrc)
    print(idst)
    # Perform the transforms.
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessImage,M,
                                 img_size,flags=cv2.INTER_LINEAR)

    warped_orig = cv2.warpPerspective(img,M,
                                      img_size,flags=cv2.INTER_LINEAR)

    write_name = './'+dir+'/dst-overlay-warped'+str(idx+1)+'.jpg'
    cv2.polylines(warped_orig, [isrc], True, (0,255,0), 3)
    cv2.polylines(warped_orig, [idst], True, (255,0,0), 3)
    cv2.imwrite(write_name, warped_orig)
    #
    # Tracking the left and right lane lines from bottom to top of the image.
    #
    window_width = 25
    window_height = 80
    # Set up th overall class to do the tracking from tracker.py
    curve_centers = tracker(Mywindow_width = window_width,
                            Mywindow_height = window_height,
                            Mymargin = 25,
                            My_ym = 10/720,
                            My_xm = 4/384,
                            Mysmooth_factor = 2)
    window_centroids = curve_centers.find_window_centroids(warped)

    # Draw points
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []
    # Go up from the bottom level.
    for level in range(0,len(window_centroids)):
        # Mask
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        
        l_mask = window_mask(window_width,window_height,
                             warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,
                             warped,window_centroids[level][1],level)
        # Add graphics
        l_points[(l_points ==255) | ((l_mask == 1))] = 255
        r_points[(r_points ==255) | ((r_mask == 1))] = 255

    # Draw 
    template = np.array(r_points+l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warp_image = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    warped_plus_tracked = cv2.addWeighted(warp_image, 1, template, 0.5, 0.0)

    # Write it out.
    print('Gradient + Color Thresholding Warp Perspective', idx, ' ', fname)
    write_name = './'+dir+'/warped-lines'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, warped)


    write_name = './'+dir+'/preprocessed'+str(idx+1)+'.jpg'
#    cv2.polylines(img, [isrc], True, (0,255,0), 3)
#    cv2.polylines(img, [idst], True, (255,0,0), 3)
    cv2.imwrite(write_name, img)

    write_name = './'+dir+'/undistorted'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, c_binary*255)

    write_name = './'+dir+'/tracked'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, warped_plus_tracked)


    # Fit the curve to lanes.
    
    yvals = range(0,warped.shape[0])
    # Arrange the y coords of the left line of each of the 9 level rows from bottom to top.
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    # Fit the x coords of the tracked center of each of the 9 level rows from bottom to top.
    left_fit = np.polyfit(res_yvals,leftx,2)
    # Recompute the x coords for yvals directly from the equation
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    # Fit the x coords of the tracked center of each of the 9 level rows from bottom to top.
    right_fit = np.polyfit(res_yvals,rightx,2)
    # Recompute the x coords for yvals directly from the equation
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)

    # 
    left_line = np.array(list(zip(np.concatenate((left_fitx-window_width/2,
                                                  left_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals,yvals[::-1]),axis=0))), np.int32)
    right_line = np.array(list(zip(np.concatenate((right_fitx-window_width/2,
                                                   right_fitx[::-1]+window_width/2), axis=0),
                                   np.concatenate((yvals,yvals[::-1]),axis=0))), np.int32)
    middle = np.array(list(zip(np.concatenate((left_fitx+window_width/2,
                                               right_fitx[::-1]-window_width/2), axis=0),
                               np.concatenate((yvals,yvals[::-1]),axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_line], color=[255,0,0])
    cv2.fillPoly(road, [right_line], color=[0,0,255])
    cv2.fillPoly(road, [middle], color=[0,255,0])
    cv2.fillPoly(road_bkg, [left_line], color=[255,255,255])
    cv2.fillPoly(road_bkg, [right_line], color=[255,255,255])

    # Convert all the drawings done in the warped image back to normal image coords.
    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.8, 0.0)

    # Conversion factor from pixel to meters.
    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    # Compute the radius from the equation directly from the left line again in meters
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,
                              np.array(leftx,np.float32)*xm_per_pix, 2)
    curveradl = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix +
                      curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])

    curve_fit_cr2 = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,
                              np.array(rightx,np.float32)*xm_per_pix, 2)
    curveradr = ((1 + (2*curve_fit_cr2[0]*yvals[-1]*ym_per_pix +
                      curve_fit_cr2[1])**2)**1.5)/np.absolute(2*curve_fit_cr2[0])

    # Calculate the camera center from the middle of the left and right lines from bottom row.
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if(center_diff <= 0):
        side_pos = 'right'

#    cv2.putText(result, 'Radius of Curvature = '+str(round(curveradl,3))+'(m)',(50,50),
#                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Radius of Curvature = L '+str(round(curveradl))+'(m) R '+str(round(curveradr))+'(m)',(50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'(m)'+side_pos+' of center',
                (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    write_name = './'+dir+'/lines'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, road)

    write_name = './'+dir+'/overlay-lines'+str(idx+1)+'.jpg'
    cv2.imwrite(write_name, result)

                
