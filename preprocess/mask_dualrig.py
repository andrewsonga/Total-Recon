# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

import cv2
import glob
import numpy as np
import pdb
import os
import shutil

from cameras import read_rtks

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
import torch
import torch.nn.functional as F
import torchvision
import sys
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)

try:
    detbase='./third_party/detectron2/'
    sys.path.insert(0,'%s/projects/PointRend/'%detbase)
    import point_rend
except:
    detbase='./third_party/detectron2_old/'
    sys.path.insert(0,'%s/projects/PointRend/'%detbase)
    import point_rend

sys.path.insert(0,'third_party/ext_utils')
from utils.io import save_vid
from util_flow import write_pfm
        
##############################################################################################################
########################################### modified by Chonghyuk Song #######################################
seqname_base=sys.argv[1]        # e.g. human-dualrig000; seqname_base + "-leftcam" or "-rightcam" becomes the actually seqname that names the folder inside database where the processed data will be stored
ishuman=sys.argv[2]             # 'y/n'
prefix=sys.argv[3]              # e.g. human-dualrig (i.e. seqname_base w.o. the video number)
isdynamic=sys.argv[4]           # 'y/n' (whether or not this is a dynamic scene)
# make another flag here that denotes whether or not this is a dynanmic scene or a static scene (if it's a static scene don't skip frame and just set mask to 0's everywhere)
# we can do the same with densepose (we should do the same with densepose, otherwise it will raise an error)
##############################################################################################################
##############################################################################################################

cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

#############################################################
################ modified by Chonghyuk Song #################                    
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.9
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.8
#############################################################
#############################################################

predictor = DefaultPredictor(cfg)

invalid_frameinds = []                                                                                          # indices where the predicted mask from either the leftcam image or rightcam image is faulty
valid_frameinds = list(range(len(sorted(glob.glob('tmp/%s/images/*.jpg'%(seqname_base + "-leftcam"))))))        # initialized as array = [0, 1, 2, ... N-1], where N = number of images in raw video
imgs_leftright=[]           # will store lists of images, each list corresponding to leftcam and rightcam
masks_leftright=[]          # will store lists of masks, each list corresponding to leftcam and rightcam
vises_leftright=[]          # will store lists of vis, each list corresponding to leftcam and rightcam
frames_leftright=[]
paths_leftright=[]

seqnames_leftright=[seqname_base + "-leftcam", seqname_base + "-rightcam"]
for seqname in seqnames_leftright:

    datadir='tmp/%s/images/'%seqname
    odir='database/DAVIS/'
    imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
    maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
    ##############################################################################################################
    ########################################### modified by Chonghyuk Song #######################################
    depthdir = '%s/DepthMaps/Full-Resolution/%s'%(odir,seqname)
    confdir = '%s/ConfidenceMaps/Full-Resolution/%s'%(odir,seqname)
    metadatadir = 'tmp/%s/metadata/'%seqname                # directory of raw metadata file containing intrinsics, cam2world (poses)
    camdir = 'cam-files/%s'%seqname_base                    # e.g. seqname_base = human-dualrig000; seqname_base + "-leftcam" or "-rightcam" becomes the actually seqname that names the folder inside database where the processed data will be stored
    ##############################################################################################################
    ##############################################################################################################

    #if os.path.exists(imgdir): shutil.rmtree(imgdir)
    #if os.path.exists(maskdir): shutil.rmtree(maskdir)
    #os.mkdir(imgdir)
    #os.mkdir(maskdir)
    
    imgs = []
    masks = []
    vises = []          # list of vis for saving images
    frames = []         # list of vis for making video
    paths = sorted(glob.glob('%s/*.jpg'%datadir))
    #for i, path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    for i, path in enumerate(paths):
        print(path)
        #print("tmp/seqname/images path: {}".format(path))               # tmp/cat-pikachiu-rgbd000/images/00000.jpg
        img = cv2.imread(path)
        h,w = img.shape[:2]
    
        # store at most 1080p videos
        scale = np.sqrt(1920*1080/(h*w))
        if scale<1:
            img = cv2.resize(img, (int(w*scale), int(h*scale)) )
        h,w = img.shape[:2]

        # resize to some empirical size
        if h>w: h_rszd,w_rszd = 1333, 1333*w//h 
        else:   h_rszd,w_rszd = 1333*h//w, 1333 
        img_rszd = cv2.resize(img,(w_rszd,h_rszd))

        # pad borders to make sure detection works when obj is out-of-frame
        pad=100
        img_rszd = cv2.copyMakeBorder(img_rszd,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
        
        #############################################################
        ################ modified by Chonghyuk Song #################
        # modified in order to accomodate static scenes such as SCANNet
        mask_rszd_otherobj = np.zeros((h_rszd+pad*2,w_rszd+pad*2))

        # dynamic scene    
        if isdynamic=='y':
            # pointrend
            outputs = predictor(img_rszd)
            outputs = outputs['instances'].to('cpu')
            
            mask_rszd = np.zeros((h_rszd+pad*2,w_rszd+pad*2))
            
            for it,ins_cls in enumerate(outputs.pred_classes):
                print(ins_cls)
                #if ins_cls ==15: # cat
                #if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):

                if ishuman=='y':
                    if ins_cls ==0:
                        mask_rszd += np.asarray(outputs.pred_masks[it])                 # an array of False or True, which True representing pixels with semantic label
                    
                    #############################################################
                    ################ modified by Chonghyuk Song #################
                    elif ins_cls >= 14 and ins_cls <= 23:
                        mask_rszd_otherobj += np.asarray(outputs.pred_masks[it])        # an array of False or True, which True representing pixels with semantic label
                    #############################################################
                    #############################################################

                else:
                    if ins_cls >= 14 and ins_cls <= 23:
                        mask_rszd += np.asarray(outputs.pred_masks[it])                 # an array of False or True, which True representing pixels with semantic label

                    #############################################################
                    ################ modified by Chonghyuk Song #################
                    elif ins_cls ==0:
                        mask_rszd_otherobj += np.asarray(outputs.pred_masks[it])        # an array of False or True, which True representing pixels with semantic label
                    #############################################################
                    #############################################################


            nb_components, output, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_rszd.astype(np.uint8), connectivity=8)
            if nb_components>1:
                max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])       # max_label = 1
                mask_rszd = output == max_label

            mask_rszd = mask_rszd.astype(bool).astype(int)
            mask_rszd_otherobj = mask_rszd_otherobj.astype(bool).astype(float)      # mask_rszd_otherobj needs to be of type float to be processed by cv2.reszie w/o error (mask_rszd gets changed into float via np.concatenate method)

            is_invalidframe = False

            print("mask_rszd.sum(): {}".format(mask_rszd.sum()))
            if (mask_rszd.sum())<1000: 
                #continue
                '''
                # for synchronized dropping of frames between left and right cameras
                if i not in invalid_frameinds:
                    invalid_frameinds.append(i)
                '''
                is_invalidframe = True
            
        # static scene (e.g. ScanNet)
        elif isdynamic == "n":
            mask_rszd = np.zeros_like(img_rszd)[..., 0]
            # adding this dummy entry so as to not induce a error in autogen.py and vidbase.py
            # when calling the following line of code: mask = mask/np.sort(np.unique(mask))[1] 
            mask_rszd[pad,pad] = 1.                               # this will later be negated via binary erosion (mask = binary_erosion(mask,iterations=2))
            mask_rszd = mask_rszd.astype(bool).astype(int)
        
        # instead of dropping frames with imperfect or even non-existent masks
        # we shall instead populate mask_rszd with entries of 255
        img_rszd                   = img_rszd[pad:-pad,pad:-pad]
        
        if is_invalidframe:
            mask = 255 * np.ones_like(img)                                          # shape = (h, w, 3)
            # (make sure to add a dummy entry of 1. [pad, pad]) so as not to induce an error in autogen.py and vidbase.py      
            mask[0, 0, 0] = 1.
        else:
            mask_rszd                 = mask_rszd[pad:-pad,pad:-pad]
            #outputs.pred_masks=outputs.pred_masks[:,pad:-pad,pad:-pad]
            #outputs.pred_boxes.tensor[:,:2] -= pad
            mask_rszd = np.concatenate([mask_rszd[:,:,np.newaxis]* 128,
                                        np.zeros((h_rszd, w_rszd, 1)),
                                        np.zeros((h_rszd, w_rszd, 1))],-1)          # shape = (h_rszd, w_rszd, 3)
            mask = cv2.resize(mask_rszd,(w,h))                                      # shape = (h, w, 3)

            #############################################################
            ################ modified by Chonghyuk Song #################
            mask_rszd_otherobj = mask_rszd_otherobj[pad:-pad,pad:-pad]          # shape = (h_rszd, w_rszd)
            mask_otherobj = cv2.resize(mask_rszd_otherobj, (w,h))               # shape = (h, w)

            # overlaying the "otherobj" mask onto the original mask, but only for zero-valued pixels        
            mask[(mask_otherobj > 0) & (mask[..., 0] == 0)] = 254
            #############################################################
            #############################################################

        #cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
        #cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask)
        
        if isdynamic=='y':
            # vis (doesn't actually be seemed to be used in the dataloader)
            outputs.pred_masks=outputs.pred_masks[:,pad:-pad,pad:-pad]
            outputs.pred_boxes.tensor[:,:2] -= pad

            v = Visualizer(img_rszd, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
            #outputs.remove('pred_masks')

            vis = v.draw_instance_predictions(outputs)
            vis = vis.get_image()
            
            vises.append(vis)
            #frames.append(vis[:,:,::-1])
            #cv2.imwrite('%s/vis-%05d.jpg'%(maskdir,counter), vis)
    
        imgs.append(img)
        masks.append(mask)
    
        #############################################################
        #############################################################

        ##############################################################################################################
        ########################################### modified by Chonghyuk Song #######################################
        # move the depthmap and confmap from tmp/seqname/depths, tmp/seqname/confs to 
        #print("tmp/seqname/images path: {}".format(path))               # tmp/cat-pikachiu-rgbd000/images/00000.jpg    

    imgs_leftright.append(np.stack(imgs, axis = 0))             # appending a numpy array of shape (N, H, W, 3)
    masks_leftright.append(np.stack(masks, axis = 0))           # appending a numpy array of shape (N, H, W, 3)
    if isdynamic=='y':
        vises_leftright.append(np.stack(vises, axis = 0))       # appending a numpy array of shape (N, H, W, 3)
        #frames_leftright.append(np.stack(frames, axis = 0))    # appending a numpy array of shape (N, H, W, 3)
    paths_leftright.append(paths)                               # appending a sorted list of image paths

for invalid_frameind in invalid_frameinds:
    valid_frameinds.remove(invalid_frameind)                    # we keep all frames by making "invalid_frameind" empty (instead will fill the mask up with entries of 255)

# freeing up memory
imgs=[]
masks=[]
vises=[]

# extracting valid frames
imgs_leftright = [imgs[valid_frameinds, ...] for imgs in imgs_leftright]
masks_leftright = [masks[valid_frameinds, ...] for masks in masks_leftright]
paths_leftright = [[paths[valid_frameind] for valid_frameind in valid_frameinds] for paths in paths_leftright]
if isdynamic=='y':
    vises_leftright = [vises[valid_frameinds, ...] for vises in vises_leftright]
    #frames_leftright = [frames[valid_frameinds, ...] for frames in frames_leftright]

# For valid frames: save img, mask, vis, camera poses and copy depth, conf 
for j, seqname in enumerate(seqnames_leftright):

    datadir='tmp/%s/images/'%seqname
    odir='database/DAVIS/'
    imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
    maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
    ##############################################################################################################
    ########################################### modified by Chonghyuk Song #######################################
    depthdir = '%s/DepthMaps/Full-Resolution/%s'%(odir,seqname)
    confdir = '%s/ConfidenceMaps/Full-Resolution/%s'%(odir,seqname)
    metadatadir = 'tmp/%s/metadata/'%seqname                # directory of raw metadata file containing intrinsics, cam2world (poses)
    camdir = 'cam-files/%s'%seqname_base                    # e.g. seqname_base = human-dualrig000; seqname_base + "-leftcam" or "-rightcam" becomes the actually seqname that names the folder inside database where the processed data will be stored
    ##############################################################################################################
    ##############################################################################################################

    # images, masks, paths, vis, frames, camera poses for valid frames
    '''
    imgs_valid = imgs_leftright[j][valid_frameinds, ...]                # (N_valid, H, W, 3)
    masks_valid = masks_leftright[j][valid_frameinds, ...]              # (N_valid, H, W, 3)
    paths_valid = [paths_leftright[j][valid_frameind] for valid_frameind in valid_frameinds]
        vises_valid = vises_leftright[j][valid_frameinds, ...]          # (N_valid, H, W, 3)
        frames_valid = frames_leftright[j][valid_frameinds, ...]        # (N_valid, H, W, 3)
    '''
    if isdynamic=='y':
        save_vid('%s/vis'%maskdir, vises_leftright[j][:,:,:,::-1], suffix='.mp4')
        save_vid('%s/vis'%maskdir, vises_leftright[j][:,:,:,::-1], suffix='.gif')

    world2cam_rtks = read_rtks(metadatadir, depthdir, confdir, recenter=False)                  # if recenter=False, depthdir, confdir aren't actually used inside "read_rtks"          
    world2cam_rtks = world2cam_rtks[valid_frameinds, ...]                                       # (N, 4, 4) -> (N_valid, 4, 4)

    # saving images, masks, vis, camera poses for valid frames
    # and copying depth and conf for valid frames
    for counter, (img_counter, mask_counter, vis_counter, world2cam_rtk_counter, path_counter) in enumerate(zip(imgs_leftright[j], masks_leftright[j], vises_leftright[j], world2cam_rtks, paths_leftright[j])):
        
        # 1. save image
        cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img_counter)      
        # 2. save mask          
        cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask_counter)              
        # 3. save vis
        if isdynamic=='y':
            cv2.imwrite('%s/vis-%05d.jpg'%(maskdir,counter), vis_counter)       
        
        # 4. save camera pose
        world2cam_rtk_counter = world2cam_rtk_counter.tolist()          
        # seqname = seqname_base + "-leftcam" or "-rightcam"
        # camera is saved to "cam-files/$seqname_base/$seqname-%05d%j.txt"
        print("saving camera to {}".format(os.path.join(camdir, '%s-%05d.txt'%(seqname, counter))))

        if not os.path.exists(camdir):
            os.makedirs(camdir)

        with open(os.path.join(camdir, '%s-%05d.txt'%(seqname, counter)), 'w') as f:
            for row in world2cam_rtk_counter:
                f.write(" ".join([str(x) for x in row]) + "\n")
        
        # 5. copy depth
        depth_path_tmp = path_counter.replace(".jpg", ".depth")
        depth_path_tmp = depth_path_tmp.replace("images", "depths")
        os.system("cp {} {}".format(depth_path_tmp, '%s/%05d.depth'%(depthdir,counter)))
        
        # 6. copy conf
        conf_path_tmp = path_counter.replace(".jpg", ".conf")
        conf_path_tmp = conf_path_tmp.replace("images", "confs")
        os.system("cp {} {}".format(conf_path_tmp, '%s/%05d.conf'%(confdir,counter)))