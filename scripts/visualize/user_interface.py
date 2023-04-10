import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def correspondenceGUI(leftcam_image, rightcam_image, leftcam_conf, rightcam_conf):
    # INPUTS:
    # 1. leftcam_image:     shape = (H, W, 3)
    # 2. rightcam_image:    shape = (H, W, 3)
    # 3. leftcam_conf:      shape = (H, W)
    # 4. rightcam_conf:     shape = (H, W)
    #
    # RETURNS:
    # 1. xy_coords_leftcam:     shape = (N, 2)
    # 2. xy_coords_rightcam:    shape = (N, 2)
    
    matplotlib.use('TkAgg')
    leftright_image = np.concatenate([leftcam_image, rightcam_image], axis = 1)         # shape = (H, 2W, 3)
    
    f, ax = plt.figure(figsize=(12, 8))
    ax.imshow(leftright_image)
    ax.set_title('Select at least 6 pairs of corresponding points in the left and right images')
    ax.set_axis_off()

    xy_coords_leftcam = []
    xy_coords_rightcam = []

    while True:
        #plt.sca(ax1)
        # one pair
        xy_coords = plt.ginput(2, mouse_stop=3)[0]          # a list of tuples of (x,y) coordinates (mouse_stop = 3 refers to RIGHTCLICK to exit)
        xy_coords = np.array(xy_coords).astype(np.int)      # array.shape = (N = 2, 2)

        # check whether the gt depth of the corresponding points of the rightcam image has conf. score of 2
        xy_coord_leftcam = xy_coords[0, ...]
        leftcam_conf_at_xy = leftcam_conf[xy_coord_leftcam[1], xy_coord_leftcam[0]]

        if leftcam_conf_at_xy < 1.5:
            print("this point doesn't have a high confidence, choose again")
            continue
        else:
            ax.plot(xy_coords[:, 0], xy_coords[:, 0], '*', MarkerSize=6, linewidth=2)
        
        xy_coords_leftcam.append(xy_coords[0, ...])
        xy_coords_rightcam.append(xy_coords[1, ...])

    return np.stack(xy_coords_leftcam, axis = 0), np.stack(xy_coords_rightcam, axis = 0)