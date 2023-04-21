# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

import os
import errno
from typing import Any, Dict, List, Tuple, Union
import cv2
import pdb
import configparser
import torch
import numpy as np
import imageio
import trimesh
import glob
import matplotlib.cm
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

import sys
sys.path.insert(0,'third_party')
import dataloader.vidbase as base_data
from ext_utils.flowlib import flow_to_image
from utils.colors import label_colormap

########################################################################################
############################# modified by Chonghyuk Song ###############################        
def depth_to_image(depth):
    inverted_normalized_depth = 1. - (depth-depth.min())/(depth.max()-depth.min())
    colormap = matplotlib.cm.get_cmap('magma')
    image = colormap(inverted_normalized_depth[..., 0].cpu().numpy())[:, :, :3]

    return image
########################################################################################
########################################################################################

def error_to_image(error, conf_gt = None):
    colormap = matplotlib.cm.get_cmap('viridis')
    image = colormap(error)[:, :, :3]

    if conf_gt is not None:
        image = image + (1 - image) * (conf_gt[:, :, np.newaxis] < 1.5)

    return image

def draw_lines(img, xy1s, xy2s):
    device = img.device
    colormap = label_colormap()
    len_colormap = colormap.shape[0]
    img = img.permute(1,2,0).cpu().numpy()*255
    img = img.astype(np.uint8)[:,:,::-1].copy()
    for i in range(len(xy1s)):
        color = tuple([int(x) for x in colormap[i%len_colormap]])
        p1 = tuple(xy1s[i].detach().cpu().numpy())
        p2 = tuple(xy2s[i].detach().cpu().numpy())
        cv2.circle(img,p1,3,  color)
        cv2.circle(img,p2,3,  color)
        cv2.line(img, p1, p2, color, thickness=1)
    #pdb.set_trace()
    #cv2.imwrite('tmp/0.png', img)
    #img = torch.Tensor(img).to(device).permute(2,0,1)[None]
    return img

def draw_pts(img, xys):
    device = img.device
    img = img.permute(1,2,0).cpu().numpy()*255
    img = img.astype(np.uint8)[:,:,::-1].copy()
    for point in xys:
        point = point.detach().cpu().numpy()
        cv2.circle(img,tuple(point),1,(0,0,255))
    #pdb.set_trace()
    #cv2.imwrite('tmp/0.png', img)
    #img = torch.Tensor(img).to(device).permute(2,0,1)[None]
    return img


def save_bones(bones, len_max, path):
    B = len(bones)
    elips_list = []
    elips = trimesh.creation.uv_sphere(radius=len_max/20,count=[16, 16])
    # remove identical vertices
    elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
    N_elips = len(elips.vertices)
    for bone in bones:
        center = bone[None,:3]
        orient = bone[3:7] # real first
        orient = orient / np.linalg.norm(orient, 2,-1)
        orient = R.from_quat(orient).as_matrix() # real first
        orient = orient.T # transpose R
        scale =  np.exp(bone[None, 7:10])

        elips_verts = elips.vertices
        elips_verts = elips_verts / scale
        elips_verts = elips_verts.dot(orient)
        elips_verts = elips_verts+center
        elips_list.append( trimesh.Trimesh(vertices = elips_verts, 
                                                   faces=elips.faces) )
    elips = trimesh.util.concatenate(elips_list)
    
    colormap = label_colormap()[:B]
    colormap= np.tile(colormap[:,None], (1,N_elips,1)).reshape((-1,3))
    elips.visual.vertex_colors[:len(colormap),:3] = colormap
    elips.export(path)

def vis_match(results, masks, imgs, bs,img_size,ndepth):
    # show error images
    bs = imgs.shape[0]
    for i in range(bs):
        mask_rszd = F.interpolate(masks[None],(img_size,img_size))[0,i].bool()
        img_rszd =  F.interpolate(imgs       ,(img_size,img_size))[i].permute(1,2,0)
        img_mskd = img_rszd[mask_rszd].cpu().numpy()
        
        if 'feat_err' in results.keys():
            feat_errs = results['feat_err'] 
            feat_err = feat_errs[i].view(img_size,img_size)
            feat_err[~mask_rszd] = 0.
            med = feat_err[mask_rszd].median()
            print('%d-median:%f' %(i,med))
            cv2.imwrite('tmp/match_err-%d.png'%i, (feat_err/med).cpu().numpy()*128)

    # draw lines
    if 'xyz_camera_vis' in results.keys() and 'pts_exp_vis' in results.keys():
        mask_rszd = F.interpolate(masks[None],(img_size,img_size))[0,0].bool()
        img_rszd =  F.interpolate(imgs       ,(img_size,img_size))[0].permute(1,2,0)
        xyz_coarse_frame = results['xyz_camera_vis'] 
        color_plane = torch.stack([img_rszd, torch.ones_like(img_rszd)],0).view(-1,3)
        color_plane = color_plane.cpu().numpy()
        near_plane= xyz_coarse_frame.view(bs,-1,ndepth,3)[0,:,0]
        d_near = near_plane[:,2].mean()
        near_plane[...,-1] -= d_near*0.01
        far_plane = xyz_coarse_frame.view(bs,-1,ndepth,3)[0,:,-1]
        nf_plane = torch.cat([near_plane, far_plane],0)
        #trimesh.Trimesh(nf_plane.cpu().numpy(), vertex_colors=color_plane).\
        trimesh.Trimesh(near_plane.cpu().numpy(), vertex_colors=img_rszd.view(-1,3).cpu().numpy()).\
                export('tmp/match_plane.obj')

        near_plane_mskd = near_plane[mask_rszd.view(-1)].cpu()
        pts_pred = results['pts_pred_vis'] 
        pts_pred = pts_pred[0].view(img_size,img_size,3)[mask_rszd].cpu().numpy()
        draw_lines_ray_canonical(near_plane_mskd, pts_pred,img_mskd,
                                     'tmp/match_line_pred.obj')
   
        pts_exp  = results['pts_exp_vis'] 
        pts_exp  = pts_exp[0].view(img_size,img_size,3)[mask_rszd].cpu().numpy()
        draw_lines_ray_canonical(pts_pred, pts_exp,img_mskd,
                                 'tmp/match_line_exp.obj')
    
    #pts_pred_col=results['pts_pred'][0][mask_rszd].cpu().numpy()
    #pts_exp_col = results['pts_exp'][0][mask_rszd].cpu().numpy()
    #trimesh.Trimesh(pts_pred, vertex_colors=img_mskd).export('tmp/viser_pred.obj')
    #trimesh.Trimesh(pts_exp  ,vertex_colors=img_mskd).export('tmp/viser_exp.obj')

def draw_lines_ray_canonical(near_plane_mskd, pts_exp, img_mskd, path):
    colormap = label_colormap()
    len_color = len(colormap)
    meshes = []
    idx=0
    num_pts = len(near_plane_mskd)
    for i in range(0,num_pts, num_pts//50): # display 50 points
        ## only plot idx=5
        #if idx!=5:        
        #    idx+=1
        #    continue
        segment = np.stack([near_plane_mskd[i], pts_exp[i]])
        line = trimesh.creation.cylinder(0.0001, 
                segment=segment,sections=5, vertex_colors=colormap[idx%len_color])
        meshes.append(line)
        idx+=1
    meshes = trimesh.util.concatenate(meshes)
    meshes.export(path)

def merge_dict(dict_list):
    out_dict = {}
    for k in dict_list[0].keys():
        out_dict[k] = []

    for i in range(len(dict_list)):
        for k in out_dict.keys():
            out_dict[k] += dict_list[i][k]
    return out_dict

def render_root_txt(cam_dir, cap_frame):
    # read all the data
    camlist = load_root(cam_dir, cap_frame)
    # construct camera mesh
    mesh = draw_cams(camlist)
    save_dir,seqname=cam_dir.rsplit('/',1)
    mesh.export('%s/mesh-%s.obj'%(save_dir, seqname))

def load_sils(root_dir, cap_frame):
    """
    load all the imgs with
    input is ...-(00000.png)
    """
    imglist = []
    img_path = '%s*.png'%(root_dir)
    #img_path = '%s0*.png'%(root_dir)
    all_path = sorted(glob.glob(img_path))
    if cap_frame>0:
        all_path = all_path[:cap_frame]
    for idx,path in enumerate(all_path):
        img = cv2.imread(path,0)
        imglist.append(img)
    imglist = np.asarray(imglist)
    return imglist


def load_root(root_dir, cap_frame):
    """
    load all the root se(3)
    input is ...-(00000.txt)
    """
    camlist = []
    #cam_path = '%s0*.txt'%(root_dir)
    cam_path = '%s*[0-9][0-9][0-9][0-9][0-9].txt'%(root_dir)        # rtk file name is ...-(00000.txt)
    all_path = sorted(glob.glob(cam_path))
    if cap_frame>0:
        all_path = all_path[:cap_frame]
    for idx,path in enumerate(all_path):
        rtk = np.loadtxt(path)
        camlist.append(rtk)
    camlist = np.asarray(camlist)
    return camlist

def draw_cams(all_cam, color='cool', axis=True,
        color_list = None, length = 0.036):
    """
    all_cam: a list of 4x4 cameras
    """
    # scale: the scene bound
    cmap = matplotlib.cm.get_cmap(color)                # we can also pass 'plasma' into this as a colormap for depth rendering
    all_cam = np.asarray(all_cam)
    trans_norm = np.linalg.norm(all_cam[:,:3,3],2,-1)
    valid_cams = trans_norm>0
    trans_max = np.median(trans_norm[valid_cams])
    
    ################################################################
    ################## modified by Chonghyuk Song ##################
    #scale=trans_max
    scale=0.3
    ################################################################
    ################################################################
    traj_len = len(all_cam)
    cam_list = [] 
    if color_list is None:
        color_list = np.asarray(range(traj_len))/float(traj_len)
    for j in range(traj_len):
        cam_rot  = all_cam[j][:3,:3].T
        cam_tran = -cam_rot.dot(all_cam[j][:3,3:])[:,0]
    
        radius = 0.02*scale

        cam = trimesh.creation.uv_sphere(radius=radius,count=[2, 2])

        if axis:
            #TODO draw axis
            extents = np.asarray([radius*20, radius*10, radius*0.1])
            axis = trimesh.creation.axis(origin_size = radius, 
                                        origin_color = cmap(color_list[j]),
                                        axis_radius = radius* 0.3,
                                        axis_length = length)
                                        #axis_length = radius*6)
            #extents=extents)
            #axis.vertices[:,2] += radius * 5
            #cam = trimesh.util.concatenate([elips, axis])
            cam = axis

        #cam.vertices = cam.vertices + cam_tran
        cam.vertices = cam.vertices.dot(cam_rot.T) + cam_tran
        #cam.visual.vertex_colors = cmap(float(j)/traj_len)
        cam_list.append(cam)
    mesh_cam = trimesh.util.concatenate(cam_list)
    return mesh_cam

def draw_cams_pair(cam1,cam2, color='cool', axis=True,
        color_list = None):
    frame_num = cam1.shape[0]
    cam_mesh1 = draw_cams(cam1, color=color,axis=axis,color_list=color_list)
    cam_mesh2 = draw_cams(cam2, color=color,axis=axis,color_list=color_list)

    # draw line
    lines = []
    for i in range(frame_num):
        cam1_c = -cam1[i,:3,:3].T.dot(cam1[i,:3,3:])[:,0]
        cam2_c = -cam2[i,:3,:3].T.dot(cam2[i,:3,3:])[:,0]
        segment = np.stack([cam1_c, cam2_c])
        line = trimesh.creation.cylinder(0.001,segment=segment,sections=5)
        lines.append(line)
    lines = trimesh.util.concatenate(lines)
    return cam_mesh1, cam_mesh2, lines

def save_vid(outpath, frames, suffix='.gif',upsample_frame=150., fps=30,
        is_flow=False):

    """
    save frames to video
    frames:     n,h,w,1 or n.
    """

    # convert to 150 frames
    if upsample_frame<1: upsample_frame = len(frames)
    frame_150=[]
    for i in range(int(upsample_frame)):
        fid = int(i/upsample_frame*len(frames))
        frame = frames[fid]
        if is_flow:
            frame = flow_to_image(frame)
        ########################################################################################
        ############################# modified by Chonghyuk Song ###############################
        #if frame.max()<=1: 
        # made this modification to prevent black flickering frames for seqname=catpika-dualrig-fgbg001
        if frame.max()<=1 + 2e-2:
            frame=frame*255
        ########################################################################################
        ########################################################################################
        frame = frame.astype(np.uint8)
        if suffix=='.gif':
            h,w=frame.shape[:2]
            fxy = np.sqrt(4e5/(h*w))
            frame = cv2.resize(frame,None,fx=fxy, fy=fxy)
        frame_150.append(frame)
    imageio.mimsave('%s%s'%(outpath,suffix), frame_150, fps=fps)

class visObj(object):
    """
    a class for detectron2 vis
    """
    def has(self, name: str) -> bool:
        return name in self._fields
    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

def config_to_dataloader(opts, is_eval=False):
    """
    from a dict of options {seqname, batch_size, ngpu} to a pytorch dataloader
    """
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts['seqname'])    
    numvid =  len(config.sections())-1

    datalist = []
    for i in range(numvid):
        dataset = get_config_info(opts, config, 'data_%d'%i, i, is_eval=is_eval)
        datalist = datalist + dataset
    dataset = torch.utils.data.ConcatDataset(datalist)
    return dataset

def get_config_info(opts, config, name, dataid, is_eval=False):
    #def load_attr(attrs, config, dataname):
    def load_attr(attrs, config, dataname, num_objs):
        #try:attrs['datapath'] = '%s'%(str(config.get(dataname, 'datapath')))
        #except:pass
        for obj_num in range(num_objs):
            try:attrs['datapath{}'.format(obj_num)] = '%s'%(str(config.get(dataname, 'datapath{}'.format(obj_num))))
            except:pass
        try:attrs['dframe'] = [int(i) for i in config.get(dataname, 'dframe').split(',')]
        except:pass
        try:attrs['can_frame']= int(config.get(dataname, 'can_frame'))
        except:pass
        try:attrs['init_frame']=int(config.get(dataname, 'init_frame'))
        except:pass
        try:attrs['end_frame'] =int(config.get(dataname, 'end_frame'))
        except:pass
        #try:attrs['rtk_path'] =config.get(dataname, 'rtk_path')
        #except:pass
        for obj_num in range(num_objs):
            try:attrs['rtk_path{}'.format(obj_num)]=config.get(dataname, 'rtk_path{}'.format(obj_num))
            except:pass
        return 
    
    attrs={}
    #attrs['rtk_path'] = None
    for obj_num in range(len(opts['rtk_paths'])):
        attrs['rtk_path{}'.format(obj_num)] = None

    #load_attr(attrs, config, 'data')
    #load_attr(attrs, config, name)
    load_attr(attrs, config, 'data', len(opts['rtk_paths']))
    load_attr(attrs, config, name, len(opts['rtk_paths']))

    #datapath = attrs['datapath']
    datapaths = [attrs['datapath{}'.format(obj_num)] for obj_num in range(len(opts['rtk_paths']))]

    if 'dframe' in opts.keys():
        dframe = opts['dframe'] # only in preload
    else:
        dframe = attrs['dframe']
    can_frame =attrs['can_frame']
    init_frame=attrs['init_frame']
    end_frame= attrs['end_frame']

    #rtk_path=opts['rtk_path']
    rtk_paths=opts['rtk_paths']
    numvid =  len(config.sections())-1

    if numvid==1 and not config.has_option(name, 'datapath'): 
        datapath='%s/%s'%(datapath, opts['seqname'])
    # opts rtk_path  
    for obj_num in range(len(opts['rtk_paths'])):
        rtk_path = rtk_paths[obj_num]

        if rtk_path =='':
            # rtk path from config
            rtk_path= attrs['rtk_path{}'.format(obj_num)]
        elif not os.path.isfile('%s-00000.txt'%rtk_path):
            print('loading cameras from init-cam')
            #rtk_path = '%s/%s'%(rtk_path, datapath.strip('/').split('/')[-1])
            rtk_path = '%s/%s'%(rtk_path, datapaths[obj_num].strip('/').split('/')[-1])

        rtk_paths[obj_num] = rtk_path
    
    #imglist = sorted(glob.glob('%s/*'%datapath))
    imglists = [sorted(glob.glob('%s/*'%datapath)) for datapath in datapaths]
    try: flip=int(config.get(name, 'flip'))
    except: flip=0

    if end_frame >0:
        #imglist = imglist[:end_frame]
        for obj_index, imglist in enumerate(imglists):
            imglists[obj_index] = imglist[:end_frame]

    print('init:%d, end:%d'%(init_frame, end_frame))
    # load dataset
    datasets = []
    for df in dframe:
        if 'lineload' in opts.keys() and opts['lineload']:
            # per-line loader
            #TODO
            #dataset= LineDataset(opts, imglist = imglist, can_frame = can_frame, 
            #                  dframe=df, init_frame=init_frame, 
            #                  dataid=dataid, numvid=numvid, flip=flip, is_eval=is_eval,
            #                  rtk_path=rtk_path)
            dataset= LineDataset(opts, imglists = imglists, can_frame = can_frame, 
                              dframe=df, init_frame=init_frame, 
                              dataid=dataid, numvid=numvid, flip=flip, is_eval=is_eval,
                              rtk_paths=rtk_paths)
        else:
            # per-image loader
            try:
                #dataset = VidDataset(opts, imglist = imglist, can_frame = can_frame, 
                #              dframe=df, init_frame=init_frame, 
                #              dataid=dataid, numvid=numvid, flip=flip, is_eval=is_eval,
                #              rtk_path=rtk_path)
                dataset = VidDataset(opts, imglists = imglists, can_frame = can_frame, 
                              dframe=df, init_frame=init_frame, 
                              dataid=dataid, numvid=numvid, flip=flip, is_eval=is_eval,
                              rtk_paths=rtk_paths)
            except: 
                continue
            #if rtk_path is None:
            #    dataset.has_prior_cam = False
            #else:
            #    dataset.has_prior_cam = True
            if rtk_paths[-1] is None:                   # assume that you can have prior cams for all objects or for no objects
                dataset.has_prior_cam = False
            else:
                dataset.has_prior_cam = True

            # whether to use preloaded data
            if 'preload' in opts.keys():
                dataset.preload = opts['preload']
            else:
                dataset.preload = False

        if 'multiply' in opts.keys():
            # duplicate such that it goes more than 200 iters
            dup_num = 200/(len(dataset)/opts['ngpu']/opts['batch_size'])
            if 'accu_steps' in opts.keys():
                dup_num = dup_num*opts['accu_steps']
            dup_num = int(dup_num)+1
            for i in range(dup_num):
                datasets.append(dataset)
        else:
            datasets.append(dataset)

    return datasets

class LineDataset(Dataset):
    '''
    '''

    #def __init__(self, opts, filter_key=None, imglist=None, can_frame=0,
    #                dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
    #                is_eval=False, rtk_path=None):
    def __init__(self, opts, filter_key=None, imglists=None, can_frame=0,
                    dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
                    is_eval=False, rtk_paths=None):
        super(LineDataset, self).__init__()
        self.crop_factor = 1.2
        self.imglists = imglists
        self.img_size = opts['img_size']
        #self.num_lines = (len(imglist)-1) * self.img_size # last img not saved
        self.num_lines = (len(imglists[-1])-1) * self.img_size # last img not saved

        # not used
        #seqname = imglist[0].split('/')[-2]

        #if rtk_path is not None:
        if  rtk_paths[-1] is not None:# assume that you can have prior cams for all objects or for no objects
            #self.rtklist =['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglist))]
            self.rtklists =[['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglists[-1]))] for rtk_path in rtk_paths]
        else:
            #self.rtklist =[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist]
            self.rtklists =[[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist[-1]] for _ in rtk_paths]

        # Load the annotation file.
        self.dataid = dataid
        print('%d lines' % self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        ###################################
        # multi fg-obj case (post 10/09/22)
        # to sample rays in a more equitable way across objects of all sizes

        # uniformly sample between 0,...,N-1 (where N = number of objects)
        obj2sample_index = np.random.randint(len(self.imglists))            # randomly samples an integer from [0, N)

        # sample from a multinomial distribution where the fg objects have thrice as much probability of being sampled than the bkgd
        #probs = np.ones(len(self.imglists))
        #probs[:-1] *= 3
        #probs = probs / np.sum(probs)
        #obj2sample_index = np.argmax(np.random.multinomial(1, probs))         # randomly samples an integer from [0, N)
        contains_obj2sample = False

        while not contains_obj2sample:
            try:dataid = self.dataid
            except: dataid=0
            #TODO lolalize file
            idt = index // self.img_size# idt, idy
            idy = index %  self.img_size# idt, idy

            dframe_list = [2,4,8,16,32]
            #max_id = len(self.imglist)-1
            max_id = len(self.imglists[-1])-1
            dframe_list = [1] + [i for i in dframe_list if (idt%i==0) and \
                                int(idt+i) <= max_id]
            dframe = np.random.choice(dframe_list)
            idtn = idt + dframe

            # the elem will eventually be what is used to enhouse the obj-specific datatypes such as mask_obj, dp_obj etc.
            save_dir_obj2sample = self.imglists[obj2sample_index][0].replace('JPEGImages', 'Pixels').rsplit('/',1)[0]
            data_path_obj2sample = '%s/%d_%05d/%04d.npy'%(save_dir_obj2sample, dframe, idt, idy)
            elem = np.load(data_path_obj2sample, allow_pickle=True).item()

            # check if sampled elem (line) contains a pixel belonging to "obj2sample"
            if np.any(elem['mask'] == 1) or np.mean(elem['mask']) == 255:
                contains_obj2sample = True
            else:
                # if it doesn't, resample 'index'
                index = np.random.randint(self.__len__())                   # randomly samples an integer from [0, num_lines)
        ###################################
        ###################################

        """
        try:dataid = self.dataid
        except: dataid=0
        #TODO lolalize file
        idt = index // self.img_size# idt, idy
        idy = index %  self.img_size# idt, idy

        dframe_list = [2,4,8,16,32]
        #max_id = len(self.imglist)-1
        max_id = len(self.imglists[-1])-1
        dframe_list = [1] + [i for i in dframe_list if (idt%i==0) and \
                            int(idt+i) <= max_id]
        dframe = np.random.choice(dframe_list)
        idtn = idt + dframe 

        # the elem will eventually be what is used to enhouse the obj-specific datatypes such as mask_obj, dp_obj etc.
        save_dir_bkgd = self.imglists[-1][0].replace('JPEGImages', 'Pixels').rsplit('/',1)[0]
        data_path_bkgd = '%s/%d_%05d/%04d.npy'%(save_dir_bkgd, dframe, idt, idy)
        elem = np.load(data_path_bkgd, allow_pickle=True).item()
        """

        for obj_index, (imglist, rtklist) in enumerate(zip(self.imglists, self.rtklists)):
            #save_dir  = self.imglist[0].replace('JPEGImages', 'Pixels').rsplit('/',1)[0]
            ###################################
            # multi fg-obj case (post 10/09/22)
            save_dir  = imglist[0].replace('JPEGImages', 'Pixels').rsplit('/',1)[0]
            data_path = '%s/%d_%05d/%04d.npy'%(save_dir, dframe, idt, idy)
            elem_obj = np.load(data_path, allow_pickle=True).item()
            ###################################
            ###################################

            # modify dataid according to training time ones

            # reload rtk based on rtk predictions
            # add RTK: [R_3x3|T_3x1]
            #          [fx,fy,px,py], to the ndc space
            # always forward flow
            try:
                #rtk_path = self.rtklist[idt]
                #rtk = np.loadtxt(rtk_path)
                #rtkn_path = self.rtklist[idtn]
                #rtkn = np.loadtxt(rtkn_path)
                #rtk = np.stack([rtk, rtkn])         
                rtk_path = rtklist[idt]
                rtk = np.loadtxt(rtk_path)
                rtkn_path = rtklist[idtn]
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
            
            kaug_path = '%s/%d_%05d/rtk.npy'%(save_dir, dframe, idt)
            kaug = np.load(kaug_path,allow_pickle=True).item()['kaug']
            
            #TODO fill elems
            #elem['rtk']           = rtk[None]                         # 1,2,x
            #elem['kaug']          = kaug                             
            #elem['dataid']        = np.stack([dataid, dataid])[None] 
            #elem['frameid']       = np.stack([idt,    idtn])[None]   
            #elem['lineid']        = np.stack([idy,    idy])[None]   
            elem['rtk{}'.format(obj_index)]           = rtk[None]                         # 1,2,x
            elem['kaug{}'.format(obj_index)]          = kaug   

            ###################################
            # multi fg-obj case (post 10/09/22)
            elem['mask{}'.format(obj_index)] = elem_obj['mask'].copy()
            elem['dp{}'.format(obj_index)] = elem_obj['dp'].copy()
            elem['dp_feat_rsmp{}'.format(obj_index)] = elem_obj['dp_feat_rsmp'].copy()   
            ###################################
            ###################################                       
        
        elem['dataid']        = np.stack([dataid, dataid])[None] 
        elem['frameid']       = np.stack([idt,    idtn])[None]   
        elem['lineid']        = np.stack([idy,    idy])[None]   
        return elem
    
class VidDataset(base_data.BaseDataset):
    '''
    '''
    #def __init__(self, opts, filter_key=None, imglist=None, can_frame=0,
    #                dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
    #                is_eval=False, rtk_path=None):
    def __init__(self, opts, filter_key=None, imglists=None, can_frame=0,
                    dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
                    is_eval=False, rtk_paths=None):
        super(VidDataset, self).__init__(opts, filter_key=filter_key)

        self.flip=flip
        #self.imglist = imglist
        self.imglists = imglists
        imglist = imglists[-1]
        self.imglist = imglist

        self.can_frame = can_frame
        self.dframe = dframe

        # multi-obj case
        #seqname = imglist[0].split('/')[-2]         # for now we don't care about all other pieces of data - we just care about
        seqnames = [imglist[0].split('/')[-2] for imglist in self.imglists]     # for obj-dependent pieces of data
        self.seqnames = seqnames
        self.seqname = seqnames[-1]                                                  # for obj agnostic pieces of data
        
        # multi-obj case
        #self.masklist = [i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for i in self.imglist] 
        self.masklists = [[i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for i in imglist] for imglist in self.imglists]
        self.camlist =  [i.replace('JPEGImages', 'Camera').replace('.jpg', '.txt') for i in self.imglist]
        
        #############################################################
        ################ modified by Chonghyuk Song #################
        self.deplist = [i.replace('JPEGImages', 'DepthMaps').replace('.jpg', '.depth') for i in self.imglist]
        self.conflist = [i.replace('JPEGImages', 'ConfidenceMaps').replace('.jpg', '.conf') for i in self.imglist]
        #############################################################
        #############################################################

        if dframe==1:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%self.seqname, '%s/flo-'%self.seqname) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%self.seqname, '%s/flo-'%self.seqname) for i in self.imglist]
        else:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%self.seqname, '%s/flo-'%(self.seqname)) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%self.seqname, '%s/flo-'%(self.seqname)) for i in self.imglist]

        # single-fg-obj case
        #self.featlist = [i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/feat-'%seqname) for i in self.imglist]
        #self.featlist = ['%s/feat-%05d.pfm'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in self.featlist]
        #self.bboxlist = ['%s/bbox-%05d.txt'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in self.featlist]
        #self.dplist = [i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm') for i in self.imglist]
        #self.kplist = [i.replace('JPEGImages', 'KP').replace('.jpg', '_keypoints.json').replace('.png', '_keypoints.json') for i in self.imglist]

        # multi-obj case
        self.featlists = [[i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/feat-'%seqname) for i in imglist] for (imglist, seqname) in zip(self.imglists, self.seqnames)]
        self.featlists = [['%s/feat-%05d.pfm'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in featlist] for featlist in self.featlists]
        self.bboxlists = [['%s/bbox-%05d.txt'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in featlist] for featlist in self.featlists]
        self.kplists = [[i.replace('JPEGImages', 'KP').replace('.jpg', '_keypoints.json').replace('.png', '_keypoints.json') for i in imglist] for imglist in self.imglists]
        self.dplists = [[i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm') for i in imglist] for imglist in self.imglists]

        #if rtk_path is not None:
        #    self.rtklist =['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglist))]
        #else:
        #    self.rtklist =[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist]
        
        if rtk_paths[-1] is not None:
            self.rtklists =[['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglists[obj_index]))] for obj_index, rtk_path in enumerate(rtk_paths)]
        else:
            self.rtklists =[[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglists[obj_index]] for obj_index in range(len(rtk_paths))]

        self.baselist = [i for i in range(len(self.imglist)-self.dframe)] +  [i+self.dframe for i in range(len(self.imglist)-self.dframe)]
        self.directlist = [1] * (len(self.imglist)-self.dframe) +  [0]* (len(self.imglist)-self.dframe)
        
        # to skip frames
        self.odirectlist = self.directlist.copy()
        len_list = len(self.baselist)//2
        self.fw_list = self.baselist[:len_list][init_frame::self.dframe]
        self.bw_list = self.baselist[len_list:][init_frame::self.dframe]
        self.dir_fwlist = self.directlist[:len_list][init_frame::self.dframe]
        self.dir_bwlist = self.directlist[len_list:][init_frame::self.dframe]

        if is_eval:
            self.baselist = self.fw_list
            self.directlist = self.dir_fwlist
        else:
            self.baselist = self.fw_list + self.bw_list
            self.directlist = self.dir_fwlist + self.dir_bwlist
            self.baselist =   [self.baselist[0]]   + self.baselist   + [self.baselist[-1]]
            self.directlist = [self.directlist[0]] + self.directlist + [self.directlist[-1]]

            fac = (opts['batch_size']*opts['ngpu']*200)//len(self.directlist) // numvid
            if fac==0: fac=1
            self.directlist = self.directlist*fac
            self.baselist = self.baselist*fac

        # Load the annotation file.
        self.num_imgs = len(self.directlist)
        self.dataid = dataid
        print('%d pairs of images' % self.num_imgs)

def str_to_frame(test_frames, data_info):
    if test_frames[0]=='{':
        # render a list of videos
        idx_render = []
        for i in test_frames[1:-1].split(','):
            vid_idx = int(i)
            idx_render += range(data_info['offset'][vid_idx]-vid_idx, 
                                data_info['offset'][vid_idx+1]-vid_idx-1)
    else:
        test_frames = int(test_frames)
        if test_frames==0: 
            test_frames = data_info['len_evalloader']-1
        # render specific number of frames
        idx_render = np.linspace(0,data_info['len_evalloader']-1,
                               test_frames, dtype=int)
    return idx_render

def extract_data_info(loader):
    data_info = {}
    dataset_list = loader.dataset.datasets
    data_offset = [0]
    #single-fg-obj version (assumes the datasets used for all objects are the same)
    """
    impath = []
    for dataset in dataset_list:
        impath += dataset.imglist
        data_offset.append(len(dataset.imglist))
    data_info['offset'] = np.asarray(data_offset).cumsum()
    data_info['impath'] = impath
    """
    #mult-fg-obj version (assumes the datasets for each object can be different)
    impath_objs = {}
    for obj_index in range(len(dataset_list[-1].imglists)):
        impath_objs[obj_index] = []

    for dataset in dataset_list:
        for obj_index, imglist in enumerate(dataset.imglists):
            impath_objs[obj_index] += imglist
        data_offset.append(len(dataset.imglist))
    data_info['offset'] = np.asarray(data_offset).cumsum()
    
    for obj_index, imglist in enumerate(dataset.imglists):
        data_info['impath_obj{}'.format(obj_index)] = impath_objs[obj_index]
    data_info['len_evalloader'] = len(loader)
    return data_info

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_vertex_colors(model, mesh, frame_idx=0, view_dir=None):
    # assign color to mesh verts according to current frame
    xyz_query = torch.cuda.FloatTensor(mesh.vertices, device=model.device)
    xyz_embedded = model.embedding_xyz(xyz_query) # (N, embed_xyz_channels)
    # use env code of the first frame
    env_code = model.env_code(torch.Tensor([frame_idx]).long().to(model.device))
    env_code = env_code.expand(xyz_query.shape[0],-1)
    if view_dir is None:
        # use view direction of (0,0,-1)
        dir_query = torch.zeros_like(xyz_query) 
        dir_query[:,2] = -1
    else:
        dir_query = F.normalize(view_dir, 2,-1)
    dir_embedded = model.embedding_dir(dir_query) # (N, embed_xyz_channels)
    xyz_embedded = torch.cat([xyz_embedded, dir_embedded, env_code],-1)
    #xyz_embedded = torch.cat([xyz_embedded, env_code],-1)
    vis = model.nerf_coarse(xyz_embedded)[:,:3].cpu().numpy()
    vis = np.clip(vis, 0, 1)
    return vis
