# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

import os, sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
import skimage.measure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def calculate_psnr(img1, img2, mask):
    # img1: shape = (H, W, 3), range = [0, 1]
    # img2: shape = (H, W, 3), range = [0, 1]
    # mask: shape = (H, W, 3), binary mask (only 0's and 1's), with 3 channels of identical masks

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    # img1: shape = (H, W, 3), range = [0, 1]
    # img2: shape = (H, W, 3), range = [0, 1]
    # mask: shape = (H, W, 3), binary mask (only 0's and 1's), with 3 channels of identical masks

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    #_, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    _, ssim_map = structural_similarity(img1, img2, channel_axis=-1, full=True)
   
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid

#########################################################
#################### 4. Evaluation ######################
# 1. perceptual metrics
# a) PSNR
def compute_psnr(rgb_gt, rgb, mask=None):
    # IMPORTANT!!! Both rgb_gt and rgb need to be in range [0, 1]
    # INPUTS:
    # 1. rgb_gt:            Ground truth image:                                         numpy array of shape = (H, W, 3)         
    # 2. rgb:               Rendered image:                                             numpy array of shape = (H, W, 3)
    # 3. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    #
    # OUTPUTS:
    # 1. psnr:

    if mask is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)                     # shape = (H, W, 3)
    else:
        mask_rgb = np.ones_like(rgb)                                                # shape = (H, W, 3)

    psnr = calculate_psnr(rgb_gt, rgb, mask_rgb)
    return psnr

# b) SSIM
def compute_ssim(rgb_gt, rgb, mask=None):
    # IMPORTANT!!! Both rgb_gt and rgb need to be in range [0, 1]
    # INPUTS:
    # 1. rgb_gt:            Ground truth image:                                         numpy array of shape = (H, W, 3)         
    # 2. rgb:               Rendered image:                                             numpy array of shape = (H, W, 3)
    # 3. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    #
    # # OUTPUTS:
    # 1. ssim

    if mask is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)                     # shape = (H, W, 3)
    else:
        mask_rgb = np.ones_like(rgb)                                                # shape = (H, W, 3)
    
    ssim = calculate_ssim(rgb_gt, rgb, mask_rgb)
    return ssim

# c) LPIPS
#lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)
def compute_lpips(rgb_gt, rgb, lpips_model, mask=None):
    # IMPORTANT!!! Both rgb_gt and rgb need to be in range [0, 1]
    # INPUTS:
    # 1. rgb_gt:            Ground truth image:                                         numpy array of shape = (H, W, 3)         
    # 2. rgb:               Rendered image:                                             numpy array of shape = (H, W, 3)
    # 3. lpips_model:       torch lpips_model (instantiate once in the main script, and feed as input to this method):      "lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)""
    # 4. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    #
    # # OUTPUTS:
    # 1. lpips

    rgb_gt_0 = im2tensor(rgb_gt).cuda()                                                     # torch tensor of shape = (H, W, 3)
    rgb_0 = im2tensor(rgb).cuda()                                                           # torch tensor of shape = (H, W, 3)
    
    if mask is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)                     # shape = (H, W, 3)
    else:
        mask_rgb = np.ones_like(rgb)                                                # shape = (H, W, 3)
    
    mask_0 = torch.Tensor(mask_rgb[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))            # torch tensor of shape = (1, 3, H, W)
    lpips = lpips_model.forward(rgb_gt_0, rgb_0, mask_0).item()
    return lpips
    

# 2. depth / pointcloud metrics
# a) depth error (squared distance error) for a single frame
def compute_depth_error(dph_gt, dph, conf_gt, mask=None, dep_scale = 0.2):
    # INPUTS:
    # 1. dph_gt:            Ground truth depth image:                                   numpy array of shape = (H, W)
    # 2. dph:               Rendered depth image:                                       numpy array of shape = (H, W)
    # 3. conf_gt:           Ground truth depth-confidence image:                        numpy array of shape = (H, W)
    # 4. mask:              Binary spatial mask over which to compute the metric:       numpy array of shape = (H, W)
    # 5. dep_scale:         Scale used to scale the ground truth depth during training
    #
    # RETURNS:
    # 1. depth_error:       Computes squared distance error in metric space (units: m^2)

    # scale the depth error back to metric space by dividing by "dep_scale"
    depth_diff = (dph - dph_gt) / dep_scale                                        # shape = (H, W)

    if mask is None:
        mask = np.ones_like(conf_gt)                                                    # shape = (H, W)

    # compute depth error over pixels that 1) have high confidence value (conf_gt > 1.5) and 2) a mask value of 1
    depth_error = np.mean(np.power(depth_diff[(conf_gt > 1.5) & (mask == 1.)], 2))

    return depth_error

def compute_depth_acc_at_10cm(dph_gt, dph, conf_gt, mask=None, dep_scale = 0.2):
    # INPUTS:
    # 1. dph_gt:            Ground truth depth image (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 2. dph:               Rendered depth image     (scaled by "dep_scale"):                                       numpy array of shape = (H, W)
    # 3. conf_gt:           Ground truth depth-confidence image:                                                    numpy array of shape = (H, W)
    # 4. mask:              Binary spatial mask over which to compute the metric:                                   numpy array of shape = (H, W)
    # 5. dep_scale:         Scale used to scale the ground truth depth during training
    #
    # RETURNS:
    # 1. depth accuracy at 0.1m:  Computes the number of test rays estimated with 0.1m of their ground truth
    depth_diff = (dph - dph_gt) / dep_scale                                             # depth difference in meters

    if mask is None:
        mask = np.ones_like(conf_gt)                                                    # shape = (H, W)

    # compute depth accuracy @ 0.1m over pixels that 1) have high confidence value (conf_gt > 1.5) and 2) a mask value of 1
    is_depth_accurate = (np.abs(depth_diff) < 0.1)
    depth_acc_at_10cm = np.mean(is_depth_accurate[(conf_gt > 1.5) & (mask == 1.)])

    return depth_acc_at_10cm

# root mean square metric over all frames
# used for computing rms depth error (units: meters)
def rms_metric_over_allframes(list_of_metrics):
    # INPUTS:
    # 1. list_of_metrics:       list of per-frame, mean squared metrics, each corresponding to one frame (list of np.arrays)
    # 
    # RETURNS:
    # 1. rms:                   root mean square metric over all frames

    list_of_metrics = np.stack(list_of_metrics)
    
    # for frames where "mask" is empty, the metric computation returns nan, so need to remove these frames before computing average across all frames
    nonempty_frames_obj = np.logical_not(np.isnan(list_of_metrics))            # frames with valid masks

    return np.sqrt(np.mean(list_of_metrics[nonempty_frames_obj]))

# used for computing rms depth error (units: meters)
def fg_rms_metric_over_allframes(list_of_meansquared_metrics, list_of_fgmasks, list_of_confs):
    # INPUTS:
    # 1. list_of_meansquared_metrics:       list of per-frame, mean squared metrics (list of np.arrays)
    # 2. list_of_fgmasks:           list of binary fgmasks, where 1 indicates the presence of a fg
    # 3. list_of_confs:             list of depth confidence maps 
    # 
    # RETURNS:
    # 1. rms:                       root mean square metric over all fg pixels across all frames

    list_of_meansquared_metrics = np.stack(list_of_meansquared_metrics)         # shape = (N,)
    list_of_fgmasks = np.stack(list_of_fgmasks)                 # shape = (N, H, W)
    list_of_confs = np.stack(list_of_confs)                     # shape = (N, H, W)

    # compute number of pixels in each frames that 1) belongs to the fg and 2) has depth confidence > 1.5
    list_of_valid_fgpixels = (list_of_fgmasks == 1) & (list_of_confs > 1.5)             # shape = (N, H, W)
    num_of_valid_fgpixels = np.sum(np.sum(list_of_valid_fgpixels, axis=-1), axis=-1)    # shape = (N,)

    # computed a weighted average using num_of_valid_fgpixels and list_of_avg_metrics
    list_of_squared_metrics = list_of_meansquared_metrics * num_of_valid_fgpixels               # shape = (N,)

    # for frames where "mask" is empty, the metric computation returns nan, so need to remove these frames before computing average across all frames
    nonempty_frames_obj = np.logical_not(np.isnan(list_of_meansquared_metrics))         # frames with valid masks
    empty_frames_obj = np.isnan(list_of_meansquared_metrics)
    
    if not num_of_valid_fgpixels[empty_frames_obj].size==0:
        assert np.mean(num_of_valid_fgpixels[empty_frames_obj]) == 0

    # root mean squared error
    mean_squared_error = np.sum(list_of_squared_metrics[nonempty_frames_obj]) / np.sum(num_of_valid_fgpixels[nonempty_frames_obj])        
    rms = np.sqrt(mean_squared_error)

    return rms

# average metric over all frames
# used for computing average PSNR, SSIM, LPIPS, CD, and F-scores
def average_metric_over_allframes(list_of_metrics):
    # INPUTS:
    # 1. list_of_metrics:       list of metrics, each corresponding to one frame (list of np.arrays)
    # 
    # RETURNS:
    # 1. average:               average metric over all frames

    list_of_metrics = np.stack(list_of_metrics)
    
    # for frames where "mask" is empty, the metric computation returns nan, so need to remove these frames before computing average across all frames
    nonempty_frames_obj = np.logical_not(np.isnan(list_of_metrics))            # frames with valid masks

    return np.mean(list_of_metrics[nonempty_frames_obj])

# average metric over all frames
# used for computing average PSNR, SSIM, LPIPS, CD, and F-scores
def fg_average_metric_over_allframes(list_of_avg_metrics, list_of_fgmasks, list_of_confs=None):
    # INPUTS:
    # 1. list_of_avg_metrics:           list of average metrics, each corresponding to one frame (list of np.arrays)
    # 2. list_of_fgmasks:               list of binary fgmasks, where 1 indicates the presence of a fg
    # 3. list_of_confs:                 list of depth confidence maps 
    
    # 
    # RETURNS:
    # 1. fg average:                    average metric over all fg pixels over all frames

    list_of_avg_metrics = np.stack(list_of_avg_metrics)                   # shape = (N)
    list_of_fgmasks = np.stack(list_of_fgmasks)                                 # shape = (N, H, W)

    if list_of_confs is None:
        list_of_confs = 2. * np.ones_like(list_of_fgmasks)                      # shape = (N, H, W)
    else:
        list_of_confs = np.stack(list_of_confs)                                 # shape = (N, H, W)

    # compute number of pixels in each frames that 1) belongs to the fg and 2) has depth confidence > 1.5
    list_of_valid_fgpixels = (list_of_fgmasks == 1) & (list_of_confs > 1.5)             # shape = (N, H, W)
    num_of_valid_fgpixels = np.sum(np.sum(list_of_valid_fgpixels, axis=-1), axis=-1)    # shape = (N,)

    #print("list_of_valid_fgpixels.shape: {}".format(list_of_valid_fgpixels.shape))
    #print("num_of_valid_fgpixels.shape: {}".format(num_of_valid_fgpixels.shape))

    # computed a weighted average using num_of_valid_fgpixels and list_of_avg_metrics
    list_of_summed_metrics = list_of_avg_metrics * num_of_valid_fgpixels                # shape = (N,)
    #print("list_of_avg_metrics: {}".format(list_of_avg_metrics.shape))
    #print("list_of_summed_metrics: {}".format(list_of_summed_metrics.shape))

    # for frames where "mask" is empty, the metric computation returns nan, so need to remove these frames before computing average across all frames
    nonempty_frames_obj = np.logical_not(np.isnan(list_of_summed_metrics))              # frames with valid masks

    empty_frames_obj = np.isnan(list_of_summed_metrics)
    
    if not num_of_valid_fgpixels[empty_frames_obj].size==0:
        assert np.mean(num_of_valid_fgpixels[empty_frames_obj]) == 0

    # metric averaged over all fg pixels over all frames
    fg_average = np.sum(list_of_summed_metrics[nonempty_frames_obj]) / np.sum(num_of_valid_fgpixels[nonempty_frames_obj]) 

    return fg_average
#########################################################
#########################################################