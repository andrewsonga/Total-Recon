# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import numpy as np

from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import cv2
import liblzfse
import time
from scipy.ndimage import binary_erosion

from ext_utils.util_flow import readPFM
from ext_utils.flowlib import warp_flow
from nnutils.geom_utils import resample_dp

#############################################################
################ modified by Chonghyuk Song #################    
flags.DEFINE_bool('recon_bkgd', False, 'whether or not to reconstruct just the background')
#############################################################
#############################################################

#def read_depth(filepath, height=256, width=192):
def read_depth(filepath, image_size):

    #print("depth image_size: {}".format(image_size))

    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # keep depth_img the same aspect ratio as the input image_size
    num_elems = depth_img.size
    height = (np.sqrt(num_elems / np.prod(image_size)) * image_size[0]).astype(np.int32)
    width = (np.sqrt(num_elems / np.prod(image_size)) * image_size[1]).astype(np.int32)
    depth_img = depth_img.copy().reshape((height, width))

    return depth_img

#def read_conf(filepath, height=256, width=192):
def read_conf(filepath, image_size):

    #print("conf map_size: {}".format(image_size))

    with open(filepath, 'rb') as conf_fh:
        raw_bytes = conf_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    num_elems = conf_img.size
    height = (np.sqrt(num_elems / np.prod(image_size)) * image_size[0]).astype(np.int32)
    width = (np.sqrt(num_elems / np.prod(image_size)) * image_size[1]).astype(np.int32)
    conf_img = conf_img.copy().reshape((height, width))

    return conf_img

def read_json(filepath, mask):
    import json
    with open(filepath) as f: 
        maxscore=-1
        for pid in  json.load(f)['people']:
            ppose = np.asarray(pid['pose_keypoints_2d']).reshape((-1,3))
            pocc = cv2.remap(mask.astype(int), ppose[:,0].astype(np.float32),ppose[:,1].astype(np.float32),interpolation=cv2.INTER_NEAREST)
            pscore = pocc.sum()
            if pscore>maxscore: maxscore = pscore; maxpose = ppose
    return maxpose

# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader
    '''

    def __init__(self, opts, filter_key=None):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.img_size = opts['img_size']
        self.filter_key = filter_key
        self.flip=0
        #############################################################
        ################ modified by Chonghyuk Song #################
        #self.crop_factor = 1.2
        if opts['recon_bkgd']:
            self.crop_factor = 1.0
            #self.crop_factor = 3.0
            print("SELF.CROP_FACTOR = {}".format(self.crop_factor))
        else:
            self.crop_factor = 1.2
            #self.crop_factor = 1.0
            print("SELF.CROP_FACTOR = {}".format(self.crop_factor))
        #############################################################
        #############################################################
        self.load_pair = True
        self.spec_dt = 0 # whether to specify the dframe, only in preload
    
    def mirror_image(self, img, mask):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()
            
            return img_flip, mask_flip
        else:
            return img, mask

    def __len__(self):
        return self.num_imgs
    
    def read_raw(self, im0idx, flowfw,dframe):
        img_path = self.imglist[im0idx]
        img = cv2.imread(img_path)[:,:,::-1] / 255.0
        shape = img.shape
        if len(shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        mask = cv2.imread(self.masklist[im0idx],0)
        
        #############################################################
        ################ modified by Chonghyuk Song #################
        # multi fg-obj case (post 10/09/22) 
        # commenting the following line to account for cases where msk only contains values of 0 and 254, which when normalized, ends up assigning a mask of value 0 and 254
        # leading the network to optimize a mask loss w.r.t. human mask
        #mask = mask/np.sort(np.unique(mask))[1]
        
        # multi fg-obj case (post 10/09/22) 
        # mask has shape (H, W, 3)
        # mask has mask.nelements entries 
        # if the mask represents a frame with an invalid mask, it will have mask.nelements - 1 entries of 255
        if mask[mask == 255].size == mask.size - 1:
            mask[mask != 255] = 255             # we set the dummy pixel whose value was set to 1 so as not to induce an error in autogen.py and vidbase.py   
        
        #occluder = mask==255           # (10/09/22)        # according to Gengshan, occluder no longer applies to BANMo
        #mask[occluder] = 0             # (10/09/22)        # according to Gengshan, occluder no longer applies to BANMo
        #############################################################
        #############################################################

        if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
            mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            mask = binary_erosion(mask,iterations=2)

        #############################################################
        ################ modified by Chonghyuk Song #################
        # commenting this out to quickly test out multi-object
        """
        # set the depth-confidence for points that lie on the boundary of the mask as "unconfident"
        eroded_mask = binary_erosion(mask, iterations=15)
        eroded_pixels = (mask > 0).astype(float) - (eroded_mask > 0).astype(float)
        """
        
        mask = np.expand_dims(mask, 2)
        
        dep = read_depth(self.deplist[im0idx], img.shape[:2])          # shape = (height=256, width=192)
        conf = read_conf(self.conflist[im0idx], img.shape[:2])         # shape = (height=256, width=192)

        conf[np.isnan(dep)] = 0
        conf[dep == 0] = 0                                             # modified on 11/03
        dep[np.isnan(dep)] = 4                                         # max depth recorded by the LiDAR is 4m 
        assert (not np.any(np.isnan(dep)))

        #TODO: upsample depth map "dep" to imgsize
        if dep.shape[0] !=img.shape[0] or dep.shape[1]!=img.shape[1]:
            #dep = cv2.resize(dep, img.shape[:2][::-1],interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
        dep = np.expand_dims(dep, 2)

        if conf.shape[0]!=img.shape[0] or conf.shape[1]!=img.shape[1]:
            conf = cv2.resize(conf, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            conf_mask = binary_erosion(conf,iterations=2)
            conf = conf_mask * conf

        # commenting this out to quickly test out multi-object
        """
        # set the depth-confidence for points that lie on the boundary of the mask as "unconfident"
        conf[eroded_pixels > 0] = 0
        conf = np.expand_dims(conf, 2)
        """
        #############################################################
        #############################################################

        #print('mask sort:%f'%(time.time()-ss))

        # flow
        if flowfw:
            flowpath = self.flowfwlist[im0idx]
        else:
            flowpath = self.flowbwlist[im0idx]
        flowpath = flowpath.replace('FlowBW', 'FlowBW_%d'%(dframe)).\
                            replace('FlowFW', 'FlowFW_%d'%(dframe))
        try:
            flow = readPFM(flowpath)[0]
            occ = readPFM(flowpath.replace('flo-', 'occ-'))[0]
            h,w,_ = mask.shape
            oh,ow=flow.shape[:2]
            factor_h = h/oh
            factor_w = w/ow
            flow = cv2.resize(flow, (w,h))
            occ  = cv2.resize(occ, (w,h))
            flow[...,0] *= factor_w
            flow[...,1] *= factor_h
        except:
            print('warning: loading empty flow from %s'%(flowpath))
            flow = np.zeros_like(img)
            occ = np.zeros_like(mask)
        flow = flow[...,:2]
        #occ[occluder] = 0                  # according to Gengshan, occluder no longer applies to BANMo
        
        #print('flo:%f'%(time.time()-ss))

        try:
            dp = readPFM(self.dplist[im0idx])[0]
        except:
            print('error loading densepose surface')
            dp = np.zeros_like(occ)
        try:
            dp_feat = readPFM(self.featlist[im0idx])[0]
            dp_bbox =  np.loadtxt(self.bboxlist[im0idx])
        except:
            print('error loading densepose feature')
            dp_feat =  np.zeros((16*112,112))
            dp_bbox =  np.zeros((4))
        dp= (dp *50).astype(np.int32)
        dp_feat = dp_feat.reshape((16,112,112)).copy()
        
        #print('dp:%f'%(time.time()-ss))

        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        try:
            rtk_path = self.rtklist[im0idx]
            rtk = np.loadtxt(rtk_path)
        except:
            print('warning: loading empty camera')
            print(rtk_path)
            rtk = np.zeros((4,4))
            rtk[:3,:3] = np.eye(3)
            rtk[:3, 3] = np.asarray([0,0,10])
            rtk[3, :]  = np.asarray([512,512,256,256]) 

        # create mask for visible vs unkonwn
        vis2d = np.ones_like(mask)
        
        #print('rtk:%f'%(time.time()-ss))
        
        # crop the image according to mask
        #############################################################
        ################ modified by Chonghyuk Song #################
        if self.crop_factor == 1.0:
            kaug, hp0, A, B= self.compute_crop_params(np.ones_like(mask))
            #print("kaug: {}".format(kaug))                                     # kaug = [11.21875 14.96875  0.       0.     ] = [720 / 64, 960 / 64]
        else:
            kaug, hp0, A, B= self.compute_crop_params(mask)
        #############################################################
        #############################################################
        #print('crop params:%f'%(time.time()-ss))
        x0 = hp0[:,:,0].astype(np.float32)
        y0 = hp0[:,:,1].astype(np.float32)
        img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR)
        #print("np.unique(mask) right before crop: {}".format(np.unique(mask)))
        mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
        #############################################################
        ################ modified by Chonghyuk Song #################
        #dep = cv2.remap(dep,x0,y0,interpolation=cv2.INTER_LINEAR) # modified on 11/03
        dep = cv2.remap(dep,x0,y0,interpolation=cv2.INTER_NEAREST) # modified on 11/03
        conf = cv2.remap(conf,x0,y0,interpolation=cv2.INTER_NEAREST)
        #############################################################
        #############################################################
        flow = cv2.remap(flow,x0,y0,interpolation=cv2.INTER_LINEAR)
        occ = cv2.remap(occ,x0,y0,interpolation=cv2.INTER_LINEAR)
        dp   =cv2.remap(dp,   x0,y0,interpolation=cv2.INTER_NEAREST)
        vis2d=cv2.remap(vis2d.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
        
        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        #############################################################
        ################ modified by Chonghyuk Song #################
        #mask = (mask>0).astype(float)
        #print("np.unique(mask) right before after all processing: {}".format(np.unique(mask)))
        mask = np.where(mask < 254, (mask>0).astype(int), mask).astype(float)       # we don't need to worry about intermediate values betewen 0 and 254 since mask we resized with cv2.INTER_NEAREST interpolation
        #if np.array_equal(np.unique(mask),  np.array([0., 1.])):
        #    print("np.unique(mask) after all processing: {}".format(np.unique(mask)))
        #############################################################
        #############################################################
        
        #TODO transform dp feat to same size as img
        dp_feat_rsmp = resample_dp(F.normalize(torch.Tensor(dp_feat)[None],2,1),
                                               torch.Tensor(dp_bbox)[None], 
                                               torch.Tensor(kaug   )[None], 
                                               self.img_size)
        

        rt_dict = {}
        rt_dict['img']   = img     
        rt_dict['mask']  = mask
        #############################################################
        ################ modified by Chonghyuk Song #################
        rt_dict['dep']   = dep     
        rt_dict['conf']  = conf
        #############################################################
        #############################################################  
        rt_dict['flow']  = flow  
        rt_dict['occ']   = occ   
        rt_dict['dp']    = dp    
        rt_dict['vis2d'] = vis2d 
        rt_dict['dp_feat'] = dp_feat
        rt_dict['dp_feat_rsmp'] = dp_feat_rsmp
        rt_dict['dp_bbox'] = dp_bbox
        rt_dict['rtk'] = rtk
        return rt_dict, kaug, hp0, A,B
    
    def compute_crop_params(self, mask):
        #ss=time.time()

        #############################################################
        ################ modified by Chonghyuk Song #################
        # single fg-obj case (pre 10/09/22)
        #indices = np.where(mask>0) 

        # multi fg-obj case (post 10/09/22)
        # this extracts the bounding box as follows:
		# 1. for frames with valid fg masks (i.e. entries are either 0, 1, 254):    the bounding box fits the extent of the (potentiall occluded) fg object in question
		# 2. for frames with invalid fg masks (i.e. all entries for mask is 255):   the bounding box extends the entire image
        indices = np.where(((0 < mask) & (mask < 254)) | (mask == 255))
        xid = indices[1]; yid = indices[0]
        #############################################################
        #############################################################

        center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
        length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
        length = (int(self.crop_factor*length[0]), int(self.crop_factor*length[1]))
        
        #print('center:%f'%(time.time()-ss))

        maxw=self.img_size;maxh=self.img_size
        orisize = (2*length[0], 2*length[1])
        alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
        
        # intrinsics induced by augmentation: augmented to to original img
        # correct cx,cy at clip space (not tx, ty)
        if self.flip==0:
            pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
        else:
            pps  = np.asarray([-float( center[0] - length[0] ), float( center[1] - length[1]  )])
        kaug = np.asarray([alp[0], alp[1], pps[0], pps[1]])

        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        A = np.eye(3)
        B = np.asarray([[alp[0],0,(center[0]-length[0])],
                        [0,alp[1],(center[1]-length[1])],
                        [0,0,1]]).T
        hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
        hp0 = np.dot(hp0,A.dot(B))                   # image coord
        return kaug, hp0, A,B

    #############################################################
    ################ modified by Chonghyuk Song #################
    #def flow_process(self,flow, flown, occ, occn, hp0, hp1, A,B,Ap,Bp):
    def flow_process(self,flow, flown, occ, occn, vis2d, vis2dn, hp0, hp1, A,B,Ap,Bp):
    #############################################################
    #############################################################
        maxw=self.img_size;maxh=self.img_size
        # augmenta flow
        hp1c = np.concatenate([flow[:,:,:2] + hp0[:,:,:2], np.ones_like(hp0[:,:,:1])],-1) # image coord
        hp1c = hp1c.dot(np.linalg.inv(Ap.dot(Bp)))   # screen coord
        flow[:,:,:2] = hp1c[:,:,:2] - np.stack(np.meshgrid(range(maxw),range(maxh)),-1)
        
        hp0c = np.concatenate([flown[:,:,:2] +hp1[:,:,:2], np.ones_like(hp0[:,:,:1])],-1) # image coord
        hp0c = hp0c.dot(np.linalg.inv(A.dot(B)))   # screen coord
        flown[:,:,:2] =hp0c[:,:,:2] - np.stack(np.meshgrid(range(maxw),range(maxh)),-1)

        #fb check
        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        hp0 = np.stack([x0,y0],-1)  # screen coord
        #hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord

        #############################################################
        ################ modified by Chonghyuk Song #################
        occ_pred = occ > 0 # combine uncertainty with fb uncertainty 
        occn_pred = occn > 0  
        #############################################################
        #############################################################       

        dis = warp_flow(hp0 + flown, flow[:,:,:2]) - hp0
        dis = np.linalg.norm(dis[:,:,:2],2,-1) 
        occ = dis / self.img_size * 2
        #occ = np.exp(-5*occ)  # 1/5 img size
        occ = np.exp(-25*occ)
        occ[occ<0.25] = 0. # this corresp to 1/40 img size
        #dis = np.linalg.norm(dis[:,:,:2],2,-1) * 0.1
        #occ[occ!=0] = dis[occ!=0]

        #############################################################
        ################ modified by Chonghyuk Song #################
        occ_mask = warp_flow(vis2d.astype(float), flow[:,:,:2])<1
        occ[occ_mask] = 0 # remove out-of-frame flow
        occ[occ_pred] = 0
        #############################################################
        #############################################################

        disn = warp_flow(hp0 + flow, flown[:,:,:2]) - hp0
        disn = np.linalg.norm(disn[:,:,:2],2,-1)
        occn = disn / self.img_size * 2
        occn = np.exp(-25*occn)
        occn[occn<0.25] = 0.
        #disn = np.linalg.norm(disn[:,:,:2],2,-1) * 0.1
        #occn[occn!=0] = disn[occn!=0]

        #############################################################
        ################ modified by Chonghyuk Song #################
        occn_mask = warp_flow(vis2dn.astype(float), flown[:,:,:2])<1
        occn[occn_mask] = 0 # remove out-of-frame flow
        occn[occn_pred] = 0
        #############################################################
        #############################################################

        # ndc
        flow[:,:,0] = 2 * (flow[:,:,0]/maxw)
        flow[:,:,1] = 2 * (flow[:,:,1]/maxh)
        #flow[:,:,2] = np.logical_and(flow[:,:,2]!=0, occ<10)  # as the valid pixels
        flown[:,:,0] = 2 * (flown[:,:,0]/maxw)
        flown[:,:,1] = 2 * (flown[:,:,1]/maxh)
        #flown[:,:,2] = np.logical_and(flown[:,:,2]!=0, occn<10)  # as the valid pixels

        flow = np.transpose(flow, (2, 0, 1))
        flown = np.transpose(flown, (2, 0, 1))
        return flow, flown, occ, occn
    
    def load_data(self, index):
        #pdb.set_trace()
        #ss = time.time()
        try:dataid = self.dataid
        except: dataid=0

        im0idx = self.baselist[index]
        dir_fac = self.directlist[index]*2-1
        dframe_list = [2,4,8,16,32]
        max_id = max(self.baselist)
        dframe_list = [1] + [i for i in dframe_list if (im0idx%i==0) and \
                             int(im0idx+i*dir_fac) <= max_id]
        dframe = np.random.choice(dframe_list)
        if self.spec_dt>0:dframe=self.dframe

        if self.directlist[index]==1:
            # forward flow
            im1idx = im0idx + dframe 
            flowfw = True
        else:
            im1idx = im0idx - dframe
            flowfw = False

        rt_dict, kaug, hp0, A,B = self.read_raw(im0idx, flowfw=flowfw, 
                dframe=dframe)
        img     = rt_dict['img']  
        mask    = rt_dict['mask']
        #############################################################
        ################ modified by Chonghyuk Song #################
        dep   = rt_dict['dep']
        conf    = rt_dict['conf']
        #############################################################
        #############################################################
        flow    = rt_dict['flow']
        occ     = rt_dict['occ']
        dp      = rt_dict['dp']
        vis2d   = rt_dict['vis2d']
        dp_feat = rt_dict['dp_feat']
        dp_bbox = rt_dict['dp_bbox'] 
        rtk     = rt_dict['rtk'] 
        dp_feat_rsmp = rt_dict['dp_feat_rsmp']
        frameid = im0idx
        is_canonical = self.can_frame == im0idx
            
        #print('before 2nd read-raw:%f'%(time.time()-ss))

        if self.load_pair:
            rt_dictn,kaugn,hp1,Ap,Bp = self.read_raw(im1idx, flowfw=(not flowfw),
                    dframe=dframe)
            imgn  =    rt_dictn['img']
            maskn =    rt_dictn['mask']
            #############################################################
            ################ modified by Chonghyuk Song #################
            depn=    rt_dictn['dep']
            confn =    rt_dictn['conf']
            #############################################################
            #############################################################
            flown =    rt_dictn['flow']
            occn  =    rt_dictn['occ']
            dpn   =    rt_dictn['dp'] 
            vis2dn=    rt_dictn['vis2d']
            dp_featn = rt_dictn['dp_feat']
            dp_bboxn = rt_dictn['dp_bbox'] 
            rtkn     = rt_dictn['rtk'] 
            dp_featn_rsmp = rt_dictn['dp_feat_rsmp']
            is_canonicaln = self.can_frame == im1idx

            #print('before process:%f'%(time.time()-ss))
       
            #############################################################
            ################ modified by Chonghyuk Song #################
            #flow, flown, occ, occn = self.flow_process(flow, flown, occ, occn,
            #                            hp0, hp1, A,B,Ap,Bp)
            flow, flown, occ, occn = self.flow_process(flow, flown, occ, occn,
                                        vis2d, vis2dn, hp0, hp1, A,B,Ap,Bp)
            #############################################################
            #############################################################
            
            #print('after process:%f'%(time.time()-ss))
            
            # stack data
            img = np.stack([img, imgn])
            mask= np.stack([mask,maskn])
            #############################################################
            ################ modified by Chonghyuk Song #################
            dep = np.stack([dep, depn])
            conf = np.stack([conf, confn])
            #############################################################
            #############################################################
            flow= np.stack([flow, flown])
            occ = np.stack([occ, occn])
            dp  = np.stack([dp, dpn])
            vis2d= np.stack([vis2d, vis2dn])
            dp_feat= np.stack([dp_feat, dp_featn])
            dp_feat_rsmp= np.stack([dp_feat_rsmp, dp_featn_rsmp])
            dp_bbox = np.stack([dp_bbox, dp_bboxn])
            rtk= np.stack([rtk, rtkn])         
            kaug= np.stack([kaug,kaugn])
            dataid= np.stack([dataid, dataid])
            frameid= np.stack([im0idx, im1idx])
            is_canonical= np.stack([is_canonical, is_canonicaln])

        elem = {}
        elem['img']           =  img        # s
        elem['mask']          =  mask       # s
        #############################################################
        ################ modified by Chonghyuk Song #################
        elem['dep']           =  dep
        elem['conf']          =  conf
        #############################################################
        #############################################################    
        elem['flow']          =  flow       # s
        elem['occ']           =  occ        # s 
        elem['dp']            =  dp         # x
        elem['dp_feat']       =  dp_feat    # y
        elem['dp_feat_rsmp']  =  dp_feat_rsmp    # y
        elem['dp_bbox']       =  dp_bbox    
        elem['vis2d']         =  vis2d      # y
        elem['rtk']           =  rtk
        elem['kaug']          =  kaug
        elem['dataid']        =  dataid
        elem['frameid']       =  frameid
        elem['is_canonical']  =  is_canonical
        return elem

    def preload_data(self, index):
        #TODO combine to a single function with load_data
        try:dataid = self.dataid
        except: dataid=0

        im0idx = self.baselist[index]
        dir_fac = self.directlist[index]*2-1
        dframe_list = [2,4,8,16,32]
        max_id = max(self.baselist)
        dframe_list = [1] + [i for i in dframe_list if (im0idx%i==0) and \
                             int(im0idx+i*dir_fac) <= max_id]
        dframe = np.random.choice(dframe_list)
        if self.spec_dt>0:dframe=self.dframe

        save_dir  = self.imglist[0].replace('JPEGImages', 'Preload').rsplit('/',1)[0]
        data_path = '%s/%d_%05d.npy'%(save_dir, dframe, im0idx)
        elem = np.load(data_path,allow_pickle=True).item()
        # modify dataid according to training time ones
        elem['dataid'] = np.stack([dataid, dataid])[None]

        # reload rtk based on rtk predictions
        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        # always forward flow
        im1idx = im0idx + dframe 
        try:
            rtk_path = self.rtklist[im0idx]
            rtk = np.loadtxt(rtk_path)
            rtkn_path = self.rtklist[im1idx]
            rtkn = np.loadtxt(rtkn_path)
            rtk = np.stack([rtk, rtkn])         
        except:
            print('warning: loading empty camera')
            print(rtk_path)
            rtk = np.zeros((4,4))
            rtk[:3,:3] = np.eye(3)
            rtk[:3, 3] = np.asarray([0,0,10])
            rtk[3, :]  = np.asarray([512,512,256,256]) 
            rtkn = rtk.copy()
            rtk = np.stack([rtk, rtkn])         
        elem['rtk']= rtk[None]

        for k in elem.keys():
            elem[k] = elem[k][0]
            if not self.load_pair:
                elem[k] = elem[k][:1]
        
        # deal with img_size (only for eval visualization purpose)
        current_size = elem['img'].shape[-1]
        # how to make sure target_size is even
        # target size (512?) + 2pad = image size (512)
        target_size = int(self.img_size / self.crop_factor * 1.2 /2) * 2
        pad = (self.img_size - target_size)//2 
        for k in ['img', 'mask', 'flow', 'occ', 'dp', 'vis2d']:
            tensor = torch.Tensor(elem[k]).view(1,-1,current_size, current_size)
            tensor = F.interpolate(tensor, (target_size, target_size), 
                        mode='nearest')
            tensor = F.pad(tensor, (pad, pad, pad, pad))
            elem[k] = tensor.numpy()
        # deal with intrinsics change due to crop factor
        length = elem['kaug'][:,:2] * 512 / 2 / 1.2
        elem['kaug'][:,2:] += length*(1.2-self.crop_factor)
        elem['kaug'][:,:2] *= current_size/float(target_size)

        return elem


    def __getitem__(self, index):
        if self.preload:
            # find the corresponding fw index in the dataset
            if self.directlist[index] != 1:
                refidx = self.baselist[index]-1
                same_idx = np.where(np.asarray(self.baselist)==refidx)[0]
                index = sorted(same_idx)[0]
            try:
                # fail loading the last index of the dataset
                elem = self.preload_data(index)
            except:
                print('loading %d failed'%index)
                elem = self.preload_data(0)
        else:
            elem = self.load_data(index)    
        return elem