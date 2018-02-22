import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class tracker():

    # Constructor?
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
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
        l_sum = np.sum(warped[int(3*img_hgt/4):, :int(img_wdt/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2

        r_sum = np.sum(warped[int(3*img_hgt/4):, int(img_wdt/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img_wdt/2)
        print('Left starts ', l_center, ' Right starts ', r_center)
        
        # Add what we find to the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations.
        for level in range(1, (int)(img_hgt/window_height)):
            # Convolve
            image_layer = np.sum(warped[int(img_hgt-(level+1)*window_height):int(img_hgt-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find left centroid.
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,img_wdt))
            lmax = np.max(conv_signal[l_min_index:l_max_index])
            if(lmax > 0):
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find right centroid.
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,img_wdt))
            print(r_min_index, r_max_index)
            rmax = np.max(conv_signal[r_min_index:r_max_index])
            if(rmax > 0):
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            # Add to list.
            window_centroids.append((l_center, r_center))
            print('Left ', level, ' ', l_center, ' Right ', level, ' ', r_center, ' Conv ', rmax)

        self.recent_centers.append(window_centroids)
        # Return
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)
    


