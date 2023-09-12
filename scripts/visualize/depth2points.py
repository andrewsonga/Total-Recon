# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

from absl import flags, app
import sys
sys.path.insert(0,'')
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import pdb
import copy
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
import imageio
from collections import defaultdict

from dataloader.vidbase import read_depth, read_conf
from utils.io import save_vid, str_to_frame, save_bones, load_root, load_sils, depth_to_image
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer_objs
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, \
                                raycast, sample_xy, K2inv, get_near_far, \
                                chunk_rays
from nnutils.rendering import render_rays
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 

# script specific ones
####################################################################################################
################################### modified by Chonghyuk Song #####################################    
flags.DEFINE_integer('startframe', 0, 'number of start frame to render')
flags.DEFINE_integer('maxframe', -1, 'maximum number frame to render')
####################################################################################################
####################################################################################################
flags.DEFINE_integer('vidid', 0, 'video id that determines the env code')
flags.DEFINE_integer('bullet_time', -1, 'frame id in a video to show bullet time')
flags.DEFINE_float('scale', 0.1,
        'scale applied to the rendered image (wrt focal length)')
flags.DEFINE_float('dep_scale', 0.1,
'scale applied to gt depth')
flags.DEFINE_string('rootdir', 'tmp/traj/','root body directory')
flags.DEFINE_string('nvs_outpath', 'tmp/nvs-','output prefix')
flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
opts = flags.FLAGS

def prepare_ray_cams_wo_kaug(rtks, device):
    rtks = torch.Tensor(rtks).to(device)
    Rmat = rtks[:,:3,:3]
    Tmat = rtks[:,:3,3]
    Kinv = K2inv(rtks[:,3])

    return Rmat, Tmat, Kinv

def construct_rays_nvs(img_size, rtks, near_far, rndmask, device):
    """
    rndmask: controls which pixel to render
    """
    bs = rtks.shape[0]
    rtks = torch.Tensor(rtks).to(device)
    rndmask = torch.Tensor(rndmask).to(device).view(-1)>0

    _, xys = sample_xy(img_size, bs, 0, device, return_all=True)
    xys=xys[:,rndmask]
    Rmat = rtks[:,:3,:3]
    Tmat = rtks[:,:3,3]
    Kinv = K2inv(rtks[:,3])
    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    return rays

def main(_):

    loadname_objs = ["logdir/{}/obj0/".format(opts.seqname), "logdir/{}/obj1/".format(opts.seqname)]
    rootdir_objs = ["logdir/{}/obj0/".format(opts.seqname), "logdir/{}/obj1/".format(opts.seqname)]
    #loadname_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-e120-b256-ft3", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-alignedframes-frame380-focal800-e120-b256-ft3"]
    #rootdir_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-e120-b256-ft3/", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-alignedframes-frame380-focal800-e120-b256-ft3/"]
    
    trainer_objs = []
    rtks_objs = []
    opts_list = []

    for loadname, rootdir in zip(loadname_objs, rootdir_objs):
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.model_path = "{}/params_latest.pth".format(loadname)
        opts_obj.seqname = opts.seqname                                 # to be used for loading the appropriate config file
        opts_obj.rootdir = rootdir                                      # to be used for loading camera

        opts_obj.use_corresp = opts.use_corresp
        opts_obj.dist_corresp = opts.dist_corresp
        opts_obj.use_unc = opts.use_unc
        opts_obj.perturb = opts.perturb
        opts_obj.chunk = opts.chunk
        opts_obj.use_3dcomposite = opts.use_3dcomposite

        opts_list.append(opts_obj)

        # instantiating trainer for each object (fg / bkgd)
        trainer = v2s_trainer_objs([opts_obj], is_eval = True)
        data_info = trainer.init_dataset()
        trainer.define_model_objs(data_info)
        trainer.model.eval()

        dataset = trainer.evalloader.dataset
        trainer_objs.append(trainer)
        
        # list of gt depthmaps, gt confidence maps, gt segmentation masks
        gt_deplist = dataset.datasets[0].deplist[opts.startframe:opts.maxframe]
        gt_conflist = dataset.datasets[0].conflist[opts.startframe:opts.maxframe]
        gt_masklist = dataset.datasets[0].masklist[opts.startframe:opts.maxframe]

        # camera poses
        rtks = load_root(rootdir, 0)            # cap frame=0=>load all
        rtks[:,3] = rtks[:,3]*opts.scale        # scaling intrinsics
        rtks_objs.append(rtks)

    if opts.maxframe > 0:
        size = opts.maxframe - opts.startframe
        sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
    else:
        size = len(rtks_objs[-1]) - opts.startframe
        sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
    
    rtks_objs = [rtks[sample_idx, ...] for rtks in rtks_objs]                           # a list of [4, 4] camera matrices

    # hardcoded for now
    rndsils = load_sils("/".join(gt_masklist[0].split("/")[:-1]) + "/", 0)[sample_idx, ...]
    
    img_size = rndsils[0].shape
    if img_size[0] > img_size[1]:
        img_type='vert'
    else:
        img_type='hori'
    
    opts.render_size = img_size

    # determine render image scale
    bs = len(rtks_objs[-1])
    img_size = int(max(img_size)*opts.scale)
    opts.render_size = img_size
    print("render size: %d"%img_size)

    # iterating over objects
    vertex_colors = [np.array([1., 0., 0.]), np.array([0., 1., 0.])]

    for obj_index, (trainer, rtks) in enumerate(zip(trainer_objs, rtks_objs)):
        
        # foreground
        rndsils = load_sils("/".join(gt_masklist[0].split("/")[:-1]) + "/", 0)[sample_idx, ...]      # (N, H, W)
        
        # background
        if obj_index == len(rtks_objs) - 1:
            #rndsils = 1. - rndsils
            rndsils = np.ones_like(rndsils)

        model = trainer.model.objs[0]
        model.img_size = img_size
        nerf_models = model.nerf_models
        embeddings = model.embeddings

        vars_np = {}
        vars_np['rtk'] = rtks
        vars_np['idk'] = np.ones(bs)
        near_far = torch.zeros(bs,2).to(model.device)
        near_far = get_near_far(near_far,
                                vars_np,
                                pts=model.latest_vars['mesh_rest'].vertices)

        vidid = torch.Tensor([opts.vidid]).to(model.device).long()
        source_l = model.data_offset[opts.vidid+1] - model.data_offset[opts.vidid] -1
        embedid = torch.Tensor(sample_idx).to(model.device).long() + \
                model.data_offset[opts.vidid]
        if opts.bullet_time>-1: embedid[:] = opts.bullet_time+model.data_offset[opts.vidid]
        print(embedid)
    
        dphs = []
        dphs_gt = []
        confs_gt = []

        dph_images = []
        dphgt_images = []

        for i, frame_index in enumerate(sample_idx):
            print("synthesizing frame {}".format(i))
            
            rndsil = rndsils[i]
            rndmask = np.zeros((img_size, img_size))

            if img_type=='vert':
                size_short_edge = int(rndsil.shape[1] * img_size/rndsil.shape[0])
                rndsil = cv2.resize(rndsil, (size_short_edge, img_size))
                rndmask[:,:size_short_edge] = rndsil

            else:
                size_short_edge = int(rndsil.shape[0] * img_size/rndsil.shape[1])
                rndsil = cv2.resize(rndsil, (img_size, size_short_edge))
                rndmask[:size_short_edge] = rndsil

            # read depth
            dph_gt = read_depth(gt_deplist[i]) * opts.dep_scale
            conf_gt = read_conf(gt_conflist[i])
            
            conf_gt[np.isnan(dph_gt)] = 0
            dph_gt[np.isnan(dph_gt)] = 4                          # max depth recorded by the LiDAR is 4m 
            assert (not np.any(np.isnan(dph_gt)))

            #TODO: upsample depth map "dep" to mask size
            if dph_gt.shape[0] != rndsil.shape[0] or dph_gt.shape[1] != rndsil.shape[1]:
                dph_gt = cv2.resize(dph_gt, rndsil.shape[::-1],interpolation=cv2.INTER_LINEAR)
            dph_gt = np.expand_dims(dph_gt, 2)

            if conf_gt.shape[0]!= rndsil.shape[0] or conf_gt.shape[1]!= rndsil.shape[1]:
                conf_gt = cv2.resize(conf_gt, rndsil.shape[::-1],interpolation=cv2.INTER_NEAREST)
                conf_mask = binary_erosion(conf_gt,iterations=2)
                conf_gt = conf_mask * conf_gt
            conf_gt = np.expand_dims(conf_gt, 2)

            # read conf
            dphs_gt.append(dph_gt)
            confs_gt.append(conf_gt)

            # construct rays
            rays = construct_rays_nvs(model.img_size, rtks[i:i+1], 
                                       near_far[i:i+1], rndmask, model.device)

            # add env code
            rays['env_code'] = model.env_code(embedid[i:i+1])[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

            # add bones
            time_embedded = model.pose_code(embedid[i:i+1])[:,None]
            rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
            if opts_list[obj_index].lbs and model.num_bone_used>0:
                bone_rts = model.nerf_body_rts(embedid[i:i+1])
                rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
                model.update_delta_rts(rays)

            with torch.no_grad():
                # render images only
                results=defaultdict(list)
                bs_rays = rays['bs'] * rays['nsample'] #

                for j in range(0, bs_rays, opts.chunk):
                    rays_chunk = chunk_rays(rays,j,opts.chunk)

                    ####################################################################################################
                    ################################### modified by Chonghyuk Song #####################################    
                    #pdb.set_trace()
                    ####################################################################################################
                    ####################################################################################################

                    rendered_chunks = render_rays(nerf_models,
                                embeddings,
                                rays_chunk,
                                N_samples = opts.ndepth,
                                perturb=0,
                                noise_std=0,
                                chunk=opts.chunk, # chunk size is effective in val mode
                                use_fine=True,
                                img_size=model.img_size,
                                obj_bound = model.latest_vars['obj_bound'],
                                render_vis=True,
                                opts=opts_list[obj_index],
                                )
                    for k, v in rendered_chunks.items():
                        results[k] += [v]
            
            for k, v in results.items():
                v = torch.cat(v, 0)
                v = v.view(rays['nsample'], -1)
                results[k] = v
            
            dph = results['depth_rnd'] [...,0].cpu().numpy()
            dphtmp = np.ones((img_size, img_size))
            dphtmp[rndmask>0] = dph

            if img_type=='vert':
                dph = dphtmp[:,:size_short_edge]
            else:
                dph = dphtmp[:size_short_edge]
        
            dphs.append(dph)
            dphs_gt.append(dph_gt[..., 0])

            # normalize both the rendered depth and gt depth with same min and max
            dph_min = np.minimum(dph[rndsil>0].min(), dph_gt[rndsil>0].min())
            dph_max = np.maximum(dph[rndsil>0].max(), dph_gt[rndsil>0].max())
            dph_norm = (dph - dph_min) / (dph_max - dph_min)
            dph_gt_norm = (dph_gt - dph_min) / (dph_max - dph_min)

            dph_image = depth_to_image(torch.from_numpy(dph_norm[..., None]))        
            dphgt_image = depth_to_image(torch.from_numpy(dph_gt_norm))

            print("dph_image: {}".format(dph_image.shape))
            print("dphgt_image: {}".format(dphgt_image.shape))

            dph_image = np.repeat(rndsil[..., None] > 0, 3, axis = -1) * dph_image
            dphgt_image = np.repeat(rndsil[..., None] > 0, 3, axis = -1) * dphgt_image

            dph_images.append(dph_image*255)
            dphgt_images.append(dphgt_image*255)

            cv2.imwrite(os.path.join(opts.nvs_outpath, 'dph-obj%d_%05d.png'%(obj_index, i)), dph_image[...,::-1]*255)
            cv2.imwrite(os.path.join(opts.nvs_outpath, 'dphgt-obj%d_%05d.png'%(obj_index, i)), dphgt_image[...,::-1]*255)
        
            # backproject the depth map
            K_mat = np.eye(3)
            K_mat[0, 0] = rtks[i][3, 0]
            K_mat[1, 1] = rtks[i][3, 1]
            K_mat[0, 2] = rtks[i][3, 2]
            K_mat[1, 2] = rtks[i][3, 3]

            u, v = np.meshgrid(np.arange(0, dph_gt.shape[1]), np.arange(0, dph_gt.shape[0]))
            u = u.ravel()
            v = v.ravel()
            rndsil_ravel = rndsil.ravel()

            # backprojecting gt depth map
            dph_gt = dph_gt.ravel()
            dph = dph.ravel()
            conf_gt = conf_gt.ravel()

            u_valid = u[(conf_gt == 2.) & (rndsil_ravel>0)]
            v_valid = v[(conf_gt == 2.) & (rndsil_ravel>0)]
            pixels_homogen = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis = 0)

            dph_gt_valid = dph_gt[(conf_gt == 2.) & (rndsil_ravel>0)]
            dph_valid = dph[(conf_gt == 2.) & (rndsil_ravel>0)]
            points_gt_3d = np.transpose(np.matmul(np.linalg.inv(K_mat), pixels_homogen) * np.repeat(dph_gt_valid[None, :], 3, axis = 0))      # (N, 3)
            points_3d = np.transpose(np.matmul(np.linalg.inv(K_mat), pixels_homogen) * np.repeat(dph_valid[None, :], 3, axis = 0))            # (N, 3)

            points_gt_3d = trimesh.Trimesh(points_gt_3d, vertex_colors = vertex_colors[obj_index] * 0.5)
            points_3d = trimesh.Trimesh(points_3d, vertex_colors = vertex_colors[obj_index])
            points_gt_3d.export(os.path.join(opts.nvs_outpath, 'pointsgt-obj%d-frame%d.obj'%(obj_index, frame_index)))
            points_3d.export(os.path.join(opts.nvs_outpath, 'points-obj%d-frame%d.obj'%(obj_index, frame_index)))

        save_vid(os.path.join(opts.nvs_outpath, 'dph-obj%d'%(obj_index)), dph_images, suffix='.mp4', upsample_frame=-1, fps=10)
        save_vid(os.path.join(opts.nvs_outpath, 'dphgt-obj%d'%(obj_index)), dphgt_images, suffix='.mp4', upsample_frame=-1, fps=10)

    
if __name__ == '__main__':
    app.run(main)