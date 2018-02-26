import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Class that holds both the left and right line tracking data        
class tracker():

    # Constructor?
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin,
                 My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
        # past left right center list
        self.recent_centers = []

        # Pixel width of window
        self.window_width = Mywindow_width

        # Pixel height of window
        self.window_height = Mywindow_height

        # Margin
        self.margin = Mymargin

        # Meters per pixel in y.
        self.ym_per_pix = My_ym

        # Meters per pixel in y.
        self.xm_per_pix = My_xm

        # Smooth factor.
        self.smooth_factor = Mysmooth_factor

    # Tracking function
    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = []
        window = np.ones(window_width)

        # Sum quarter bottom of image to get slice.
        img_hgt = warped.shape[0]
        img_wdt = warped.shape[1]
        # Left
        l_sum = np.sum(warped[int(3*img_hgt/4):, :int(img_wdt/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        # Right
        r_sum = np.sum(warped[int(3*img_hgt/4):, int(img_wdt/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img_wdt/2)
        
        # Add what we find to the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations.
        for level in range(1, (int)(img_hgt/window_height)):
            # Convolve the rectangle on the layer.
            image_layer = np.sum(warped[int(img_hgt-(level+1)*window_height):int(img_hgt-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Offset to center
            offset = window_width/2
            # Find left centroid of the maximum signal.            
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,img_wdt))
            lmax = np.max(conv_signal[l_min_index:l_max_index])
            # Do not update if the signal is zero
            if(lmax > 0):
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find right centroid of the maximum signal.            
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,img_wdt))

            rmax = np.max(conv_signal[r_min_index:r_max_index])
            # Do not update if the signal is zero
            if(rmax > 0):
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            # Add to list.
            window_centroids.append((l_center, r_center))
        # Append to the list window_centroids.
        self.recent_centers.append(window_centroids)
        # Return the average over smooth_factor count of the past centers.
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
    


