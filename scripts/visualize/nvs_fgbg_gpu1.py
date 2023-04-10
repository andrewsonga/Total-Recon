# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
bash scripts/render_nvs.sh
"""
from absl import flags, app
import sys
sys.path.insert(0,'')
sys.path.insert(0,'third_party')
import torch
import numpy as np
import os
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
import glob
import pdb
import copy
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio
from collections import defaultdict
import configparser
import chamfer3D.dist_chamfer_3D
import fscore     
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from utils.io import save_vid, str_to_frame, save_bones, load_root, load_sils, depth_to_image, error_to_image
from utils.colors import label_colormap
from nnutils.train_utils_objs import v2s_trainer_objs
from nnutils.geom_utils import obj_to_cam, pinhole_cam, obj2cam_np, tensor2array, vec_to_sim3, \
                                raycast, sample_xy, K2inv, get_near_far, \
                                chunk_rays
from nnutils.rendering import render_rays_objs
from nnutils.eval_utils import im2tensor, calculate_psnr, calculate_ssim
from nnutils.eval_utils import compute_psnr, compute_ssim, compute_lpips, compute_depth_error, compute_chamfer_dist_fscore, rms_metric_over_allframes, average_metric_over_allframes
from dataloader.vidbase import read_depth, read_conf
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer

# from current director: $BANMO_DIR/scripts/visualize/
from compute_flow import flow_inference
import lpips_models

# from third_party
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
from flowutils.flowlib import point_vec
import pytorch3d
import pytorch3d.ops

# script specific ones
####################################################################################################
################################### modified by Chonghyuk Song #####################################    
#flags.DEFINE_integer('maxframe', 1, 'maximum number frame to render')
flags.DEFINE_float('startangle', -30, 'end angle for freeze nvs')
flags.DEFINE_float('endangle', 360, 'end angle for freeze nvs')
flags.DEFINE_integer('startframe', 0, 'number of start frame to render')
flags.DEFINE_integer('maxframe', -1, 'maximum number frame to render')
flags.DEFINE_integer('freeze_frame', 0, 'frame number to freeze at')
flags.DEFINE_bool('freeze', False, 'freeze object at specified frame number')
flags.DEFINE_bool('fixed_view', False, 'fix camera view')
flags.DEFINE_integer('fix_frame', -1, 'frame number to fix camera at: -1 denotes user-defined camera view')
flags.DEFINE_bool('topdown_view', False, 'render from topdown view')
flags.DEFINE_float('topdowncam_offset_y', 0.24, 'offset along the y-direction of the fgroot frame for topdown view')
flags.DEFINE_float('topdowncam_offset_x', 0., 'offset along the x-direction of the fgroot frame for topdown view')
flags.DEFINE_float('topdowncam_offset_z', 0., 'offset along the z-direction of the fgroot frame for topdown view')
flags.DEFINE_float('topdowncam_offsetabt_xaxis', 0., 'rotation offset abt x-axis of the topdowncam frame (in degrees)')
flags.DEFINE_float('topdowncam_offsetabt_yaxis', 0., 'rotation offset abt y-axis of the topdowncam frame (in degrees)')
flags.DEFINE_float('topdowncam_offsetabt_zaxis', 0., 'rotation offset abt z-axis of the topdowncam frame (in degrees)')
flags.DEFINE_bool('firstperson_view', False, 'render from firstperson view (Battleground style) of the foreground object')
flags.DEFINE_float('firstpersoncam_offset_x', 0, 'offset along the x-direction of the fgmeshcenter frame for first-person view')
flags.DEFINE_float('firstpersoncam_offset_y', 0, 'offset along the y-direction of the fgmeshcenter frame for first-person view')
flags.DEFINE_float('firstpersoncam_offset_z', 0, 'offset along the z-direction of the fgmeshcenter frame for first-person view')
flags.DEFINE_float('firstpersoncam_offsetabt_xaxis', 0,
        'offset rotation abt the x-direction of first-person camera')
flags.DEFINE_float('firstpersoncam_offsetabt_yaxis', 0,
        'offset rotation abt the y-direction of first-person camera')
flags.DEFINE_float('firstpersoncam_offsetabt_zaxis', 0,
        'offset rotation abt the z-direction of first-person camera')
flags.DEFINE_float('scale_fps', 0.75,
        'scale to intrinsics of first-person view rendering to enable larger-fov')
flags.DEFINE_integer('fg_normalbase_vertex_index', 55,
        'index of foreground mesh vertex that determines the origin and orientation of the 3d asset or egocentric camera')
flags.DEFINE_integer('fg_downdir_vertex_index', 55,
        'index of foreground mesh vertex that determines the down direction of the egocentric camera relative to the fg_normalbase_vertex')
flags.DEFINE_integer('fg_obj_index', 0,
        'index that denotes the fg object we would like to perform fps-, tps-, bev- synthesis')
flags.DEFINE_integer('asset_obj_index', 0,
        'index that denotes the fg object that we would like to attach the 3d asset to')
flags.DEFINE_bool('thirdperson_view', False, 'render from thirdperson view (Battleground style) of the foreground object')
flags.DEFINE_float('thirdpersoncam_fgmeshcenter_elevate_y', 0, 'offset along the y-direction by which we elevate the fgmeshcenter - allows one to control where on the fg mesh the 3rd person camera is focused on')
flags.DEFINE_float('thirdpersoncam_offset_x', 0, 'offset along the y-direction of the fgmeshcenter frame for third-person view')
flags.DEFINE_float('thirdpersoncam_offset_y', 0, 'offset along the y-direction of the fgmeshcenter frame for third-person view')
flags.DEFINE_float('thirdpersoncam_offset_z', 0, 'offset along the z-direction of the fgmeshcenter frame for third-person view')
flags.DEFINE_float('thirdpersoncam_offsetabt_xaxis', 0,
        'offset rotation abt the x-direction of third-person camera')
flags.DEFINE_float('thirdpersoncam_offsetabt_yaxis', 0,
        'offset rotation abt the y-direction of third-person camera')
flags.DEFINE_float('thirdpersoncam_offsetabt_zaxis', 0,
        'offset rotation abt the z-direction of third-person camera')
flags.DEFINE_float('scale_tps', 0.75,
        'scale to intrinsics of first-person view rendering to enable larger-fov')
flags.DEFINE_bool('stereo_view', False, 'render from the 2nd camera of a dual-camera rig')
flags.DEFINE_string('refcam', 'leftcam', '(= `leftcam` or `rightcam`) the reference camera whose RGBD images are used to train our fg-bgkd model')
flags.DEFINE_bool('flow_correspondence', False, 'use optical flow (vcnplus) to compute the correspondences between left and right camera images for stereo calibration - if false, user will have to specify him/herself')
flags.DEFINE_bool('filter_3d', False, 'attach a 3d asset as a filter on designated foreground object')
flags.DEFINE_float('asset_scale', 0.0003,
        'scale applied to the vertices of asset mesh')
flags.DEFINE_float('asset_offset_x', 0,
        'offset along the x-direction of the 3d asset')
flags.DEFINE_float('asset_offset_y', 0,
        'offset along the y-direction of the 3d asset')
flags.DEFINE_float('asset_offset_z', 0,
        'offset along the z-direction of the 3d asset')
flags.DEFINE_float('asset_offsetabt_xaxis', 0,
        'offset rotation abt the x-direction of the 3d asset')
flags.DEFINE_float('asset_offsetabt_yaxis', 0,
        'offset rotation abt the y-direction of the 3d asset')
flags.DEFINE_float('asset_offsetabt_zaxis', 0,
        'offset rotation abt the z-direction of the 3d asset')
flags.DEFINE_bool('render_cam', False, 'render a 3d asset of the camera to indicate the desired viewing direction')
flags.DEFINE_bool('render_cam_inputview', False, 'render the 3d asset of the camera from the inputview')
flags.DEFINE_bool('render_cam_stereoview', False, 'render the 3d asset of the camera from the stereoview')
flags.DEFINE_bool('render_cam_fixedview', False, 'render the 3d asset of the camera from the fixedview')
flags.DEFINE_bool('input_view', False, 'render from the camera trajectory from the original input video')
flags.DEFINE_bool('evaluate', True, 'computing evaluation metrics')
flags.DEFINE_bool('prealign', False, 'for prealigning using icp for evaluating non-depth supervised case')
####################################################################################################
####################################################################################################
flags.DEFINE_integer('vidid', 0, 'video id that determines the env code')
flags.DEFINE_integer('bullet_time', -1, 'frame id in a video to show bullet time')
flags.DEFINE_float('scale', 0.1,
        'scale applied to the rendered image (wrt focal length)')
flags.DEFINE_string('rootdir', 'tmp/traj/','root body directory')
flags.DEFINE_string('nvs_outpath', 'tmp/nvs-','output prefix')
flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
######################################## optical flow related ####################################
flags.DEFINE_integer('maxdisp', 256, 'maxium disparity. Only affect the coarsest cost volume size')
flags.DEFINE_integer('fac', 1, 'controls the shape of search grid. Only affect the coarse cost volume size')
flags.DEFINE_float('testres', 1, 'resolution')
flags.DEFINE_string('loadmodel', 'lasr_vcn/vcn_rob.pth', 'model path')

flags.DEFINE_multi_string('loadname_objs', 'None', 'name of folder inside \logdir to load into fg banmo object')

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

    if opts.loadname_objs == ['None']:
        # where seqname = savename
        obj_dirs = sorted(glob.glob("{}/{}/obj[0-9]".format(opts.checkpoint_dir, opts.seqname)))
        loadname_objs = ["{}/{}/obj{}/".format(opts.checkpoint_dir, opts.seqname, obj_index) for obj_index in range(len(obj_dirs))]
        rootdir_objs = ["{}/{}/obj{}/".format(opts.checkpoint_dir, opts.seqname, obj_index) for obj_index in range(len(obj_dirs))]
    else:
        # where seqname != savename (e.g. savename = $seqname-init)
        loadname_objs = ["{}/{}/".format(opts.checkpoint_dir, loadname_obj) for loadname_obj in opts.loadname_objs]
        rootdir_objs = ["{}/{}/".format(opts.checkpoint_dir, loadname_obj) for loadname_obj in opts.loadname_objs]

    #loadname_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-e120-b256-ft3", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-alignedframes-frame380-focal800-e120-b256-ft3"]
    #rootdir_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-e120-b256-ft3/", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-alignedframes-frame380-focal800-e120-b256-ft3/"]

    # WARNING: make sure when loading camera .txt files from folders containing pretrained models, only keep camera .txt files of the format $seqname-cam-%0d5.txt
    #loadname_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-depscale0p2-depwt0-e120-b256-ft2", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-frame380-depscale0p2-depwt0-eikonal2wt0p001-e120-b256-ft2"]
    #rootdir_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-depscale0p2-depwt0-e120-b256-ft2/", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-frame380-depscale0p2-depwt0-eikonal2wt0p001-e120-b256-ft2/"]
    #loadname_objs = ["logdir/andrew-dualcam000-depscale0p2-e120-b256-ft2", "logdir/andrew-dualcam-bkgd000-colmap-ds-depscale0p2-e120-b256-ft2"]
    #rootdir_objs = ["logdir/andrew-dualcam000-depscale0p2-e120-b256-ft2/", "logdir/andrew-dualcam-bkgd000-colmap-ds-depscale0p2-e120-b256-ft2/"]
    opts_list = []

    for loadname, rootdir in zip(loadname_objs, rootdir_objs):
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.model_path = "{}/params_latest.pth".format(loadname)
        opts_obj.logname = opts.seqname                                 # to be used for loading the appropriate config file
        opts_obj.seqname = opts.seqname                                 # to be used for loading the appropriate config file
        opts_obj.rootdir = rootdir                                      # to be used for loading camera

        opts_obj.use_corresp = opts.use_corresp
        opts_obj.dist_corresp = opts.dist_corresp
        opts_obj.use_unc = opts.use_unc
        opts_obj.perturb = opts.perturb
        opts_obj.chunk = opts.chunk
        opts_obj.use_3dcomposite = opts.use_3dcomposite

        opts_list.append(opts_obj)

    # any object agnostic settings (e.g. training settings) will be taken henceforth from the background opts (index = -1)

    trainer = v2s_trainer_objs(opts_list, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model_objs(data_info)

    model = trainer.model
    model.eval()
    dataset = trainer.evalloader.dataset
    gt_imglist_refcam = dataset.datasets[0].imglist
    gt_deplist_refcam = dataset.datasets[0].deplist
    gt_conflist_refcam = dataset.datasets[0].conflist
    
    # to account for multi-objs
    #gt_masklist_refcam = dataset.datasets[0].masklist
    gt_masklist_refcam_objs = [dataset.datasets[0].masklists[obj_index] for obj_index in range(len(loadname_objs))]

    #nerf_models = model.nerf_models
    #embeddings = model.embeddings

    # bs, 4,4 (R|T)
    #         (f|p)
    rtks_objs = []
    for opts_obj in opts_list:
        rtks = load_root(opts_obj.rootdir, 0)  # cap frame=0=>load all
        rtks_objs.append(rtks)
    if opts.maxframe > 0:
        #size = (opts.maxframe - opts.startframe) // (opts.skipframe) + 1
        size = opts.maxframe - opts.startframe
    else:
        #size = (len(rtks_objs[-1])-1 - opts.startframe) // (opts.skipframe) + 1
        size = len(rtks_objs[-1])-1 - opts.startframe

    # hardcoded for now
    rndsils = np.ones((size, 960, 720))             # (N, H, W)        # set opts.scale to 0.375
    img_size = rndsils[0].shape
    if img_size[0] > img_size[1]:
        img_type='vert'
    else:
        img_type='hori'

    if opts.render_cam:
        # rkts_objs used to compute the desired camera view for rendering the camera asset
        rtks_objs_rendercam = [rtks.copy() for rtks in rtks_objs]
        
        if opts.render_cam_inputview:
            if opts.maxframe > 0:
                #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
                sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
            else:
                #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
                sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
            rtks_objs_rendercam = [rtks[sample_idx, ...] for rtks in rtks_objs_rendercam]       # a list of [4, 4] camera matrices

        elif opts.render_cam_stereoview:
            if opts.maxframe > 0:
                sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
            else:
                sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
            rtks_objs_rendercam = [rtks[sample_idx] for rtks in rtks_objs_rendercam]

            # assume normrefcam2secondcam.npy is the relative transformation defined in the metric space
            refcam2secondcam = np.load("logdir/{}/normrefcam2secondcam.npy".format(opts.seqname))       # shape = (4, 4)
            
            # need to apply scaling to the translations for proper novel-view synthesis
            refcam2secondcam[:3, 3:4] *= opts_list[-1].dep_scale

            # 2. compute bkgd to novel-view transformation
            for rtks_obj_rendercam in rtks_objs_rendercam:
                # root-body pose for each object
                root2videocams_rendercam = np.tile(np.eye(4)[None, ...], (size, 1, 1))
                root2videocams_rendercam[:,:3,:] = rtks_obj_rendercam[:,:3,:]
                root2videocams_rendercam[:,3,:] = 0.
                root2videocams_rendercam[:,3,3] = 1.

                # root to novel-view transformation
                #root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
                root2novelcams_rendercam = np.matmul(refcam2secondcam, root2videocams_rendercam)
                rtks_obj_rendercam[:,:3,:] = root2novelcams_rendercam[:,:3,:]

        elif opts.render_cam_fixedview:
            if opts.maxframe > 0:
                sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
            else:
                sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)

            if opts.fix_frame >= 0:
                # camera view fixed to a specific frame's view
                rtks_objs_rendercam_fixframe = [rtks[np.repeat(opts.fix_frame, size)] for rtks in rtks_objs_rendercam]
                rtks_objs_rendercam = [rtks[sample_idx] for rtks in rtks_objs_rendercam]

            bkgd2novelcams_rendercam = np.copy(rtks_objs_rendercam_fixframe[-1])

            # extrinsics of video-view cameras (w.r.t. background)
            bkgd2videocams_rendercam = np.copy(rtks_objs_rendercam[-1])      # (N, 4, 4)
            bkgd2videocams_rendercam[:, 3, :] = 0.
            bkgd2videocams_rendercam[:, 3, 3] = 1.

            for rtks_obj_rendercam in rtks_objs_rendercam:
                # root-body pose for each object
                root2videocams_rendercam = np.tile(np.eye(4)[None, ...], (size, 1, 1))
                root2videocams_rendercam[:,:3,:] = rtks_obj_rendercam[:,:3,:]
                root2videocams_rendercam[:,3,:] = 0.
                root2videocams_rendercam[:,3,3] = 1.

                # root to novel-view transformation
                #TODO
                root2novelcams_rendercam = np.matmul(np.matmul(bkgd2novelcams_rendercam, np.linalg.inv(bkgd2videocams_rendercam)), root2videocams_rendercam)
                rtks_obj_rendercam[:,:3,:] = root2novelcams_rendercam[:,:3,:]

    ####################################################################################################
    ################################### modified by Chonghyuk Song #####################################    
    #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
    #rtks = rtks[sample_idx]
    #rndsils = rndsils[sample_idx]
    
    if opts.freeze:
        opts.nvs_outpath = opts.nvs_outpath + "-frozentime"

        bkgd2cam = np.concatenate((rtks_objs[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[-1].shape[0], axis = 0)), axis = 1)
        fg2cam = np.concatenate((rtks_objs[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
        cam2bkgd = np.linalg.inv(bkgd2cam)
        fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)      # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates  
        #mesh_rest_fg = trimesh.load('%s/mesh_rest-14.obj'%(rootdir_objs[0]),process=False)  # loading the mesh for the foreground
        mesh_rest_fg = trimesh.load('%s/mesh-rest.obj'%(rootdir_objs[0]),process=False)  # loading the mesh for the foreground
        mesh_rest_fg_center = mesh_rest_fg.vertices.mean(0)                                 # (3,)
        fgmeshcenter2fg = np.eye(4)
        fgmeshcenter2fg[:3, 3] = mesh_rest_fg_center
        fgmeshcenter2bkgd = np.matmul(fg2bkgd, np.repeat(fgmeshcenter2fg[None, :, :], fg2bkgd.shape[0], axis = 0))

        sample_idx = np.repeat(opts.freeze_frame, size)
        '''
        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]
        
        # extrinsics of video-view cameras (w.r.t. background)
        rtks_bkgd = np.copy(rtks_objs[-1])
        rtks_bkgd[:, 3, :] = 0.
        rtks_bkgd[:, 3, 3] = 1.

        # extrinsics of novel-view cameras (novel-view camera trajectory)
        bkgd2novelcams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
        for i in range(size):
            rot_turntb = cv2.Rodrigues(np.asarray([0.,(opts.startangle + i * (opts.endangle - opts.startangle) / size) * np.pi/180 ,0.]))[0]
            bkgd2novelcams[i,:3,:] = rot_turntb.dot(rtks_bkgd[i,:3,:])          # rtks_bkgd denotes the extrinsics (world2cam transformation) for the background
            bkgd2novelcams[i,:2,3] = 0
            bkgd2novelcams[i,2,3] = 0.05
        # TODO
        '''
        # let's just test it for a single object
        rtk_frozen_objs = [rtks[opts.freeze_frame].copy() for rtks in rtks_objs]
        rtk_frozen = rtk_frozen_objs[-1]
        #mesh_frozen = trimesh.load('%s/%s-mesh-%05d.obj'%(opts_obj.rootdir, 'cat-pikachiu-rgbd-bkgd000', opts.freeze_frame))
        #verts = mesh_frozen.vertices[None]        # shape = (N, 3)
        #vmean = verts[0].mean(0)
        #print("vmean: {}".format(vmean))            # coordinate of mesh mean in world-coordinate system

        ctrajs = []
        for i in range(size):
            refcam = rtk_frozen.copy()

            rot_turntb = cv2.Rodrigues(np.asarray([0.,(opts.startangle + i * (opts.endangle - opts.startangle) / size) * np.pi/180, 0.]))[0]
            refcam[:3,:3] = rot_turntb.dot(refcam[:3,:3])
            #refcam[:2,3] = 0  # trans xy
            #refcam[1,3] = 0.014
            #refcam[2,3] = 0.05
            #refcam[:2,3] = (-refcam[:3,:3].dot(vmean))[:2]
            #refcam[2,3] = -0.1
            #center = np.array([[-0.01211, -0.04408, 0.111547]]).T      # manually set based on readings from meshLab
            center = fgmeshcenter2bkgd[opts.freeze_frame, :3, 3:4]      # (3,1)

            refcam[:3,3:4] = (np.eye(3) - refcam[:3,:3]).dot(center) + refcam[:3,3:4]
            #refcam[2,3] += 0.05         # z-axis
            #refcam[1,3] += 0.05         # y-axis
            ctrajs.append(refcam)
        
        bkgd2novelcams = np.stack(ctrajs, axis = 0)

        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]
        # extrinsics of video-view cameras (w.r.t. background)
        bkgd2videocams = np.copy(rtks_objs[-1])      # (N, 4, 4)
        bkgd2videocams[:, 3, :] = 0.
        bkgd2videocams[:, 3, 3] = 1.

        for rtks_obj in rtks_objs:
            # root-body pose for each object
            root2videocams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
            root2videocams[:,:3,:] = rtks_obj[:,:3,:]
            root2videocams[:,3,:] = 0.
            root2videocams[:,3,3] = 1.

            # root to novel-view transformation
            #TODO
            root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
            #root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
            rtks_obj[:,:3,:] = root2novelcams[:,:3,:]

    elif opts.firstperson_view:
        opts.nvs_outpath = opts.nvs_outpath + "-fpsview"

        if opts.maxframe > 0:
            #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
            sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
        else:
            #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
            sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]


        # what should the novel-camera trajectory (w.r.t bkgd frame) be?
        # 
        # assuming that the fgmeshcenter frame is well aligned with the cat, we can set an offset along the xz plane of the fgmeshcenter frame and view the cat from there
        # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should a few units behind the cat in the z-direction
        bkgd2cam = np.concatenate((rtks_objs[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[-1].shape[0], axis = 0)), axis = 1)
        fg2cam = np.concatenate((rtks_objs[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
        cam2bkgd = np.linalg.inv(bkgd2cam)
        fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates  
        
        """
        mesh_rest_fg = trimesh.load('%s/mesh-rest.obj'%(rootdir_objs[0]),process=False)  # loading the mesh for the foreground
        mesh_rest_fg_center = mesh_rest_fg.vertices.mean(0)                                 # (3,)
        fgmeshcenter2fg = np.eye(4)
        fgmeshcenter2fg[:3, 3] = mesh_rest_fg_center                                        # coordinate of fgmeshcenter in fgroot frame
        firstpersoncam2fgmeshcenter = np.eye(4)
        
        firstpersoncam2fgmeshcenter[1, 3] = opts.firstpersoncam_offset_y                    # offset along the y-direction of the fgmeshcenter frame for first-person view
        firstpersoncam2fgmeshcenter[2, 3] = opts.firstpersoncam_offset_z                    # offset along the z-direction of the fgmeshcenter frame for first-person view

        fgmeshcenter2bkgd = np.matmul(fg2bkgd, np.repeat(fgmeshcenter2fg[None, :, :], fg2bkgd.shape[0], axis = 0))
        firstpersoncam2bkgd = np.matmul(fgmeshcenter2bkgd, np.repeat(firstpersoncam2fgmeshcenter[None, :, :], fg2bkgd.shape[0], axis = 0))
        
        bkgd2novelcams = np.linalg.inv(firstpersoncam2bkgd)
        # change direction of x-axis and y-axis (rows) to make them consistent with camera (NOT exactly sure why we have to do this but oh well)
        bkgd2novelcams[:, :2, :] *= -1
        """

        meshdir_fg = opts_list[opts.fg_obj_index].rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
        novelcams2fg = []
        for i, frame_idx in enumerate(sample_idx):
            
            # loading the foreground mesh (choice of foreground number represented by opts.fg_obj_index) 
            meshdir_fg_time = glob.glob(meshdir_fg + "*-mesh-%05d.obj"%(frame_idx))
            assert(len(meshdir_fg_time) == 1)
            mesh_fg_time = trimesh.load(meshdir_fg_time[0], process=False)

            # placeholder for novel-camera pose in fg frame
            novelcam2fg = np.eye(4)                                                                                
            
            # rotation component
            # extracting Z-axis direction of novel-camera frame = vertex normal
            novelcam2fg[:3, 2] = mesh_fg_time.vertex_normals[opts.fg_normalbase_vertex_index, :].copy()            # unit length normal vector

            # extracting X-axis = (downward vector in YZ plane = "y_prime") x Z-axis basis vector
            y_prime = mesh_fg_time.vertices[opts.fg_downdir_vertex_index, :] - mesh_fg_time.vertices[opts.fg_normalbase_vertex_index, :]
            yprime_cross_z = np.cross(y_prime, novelcam2fg[:3, 2])
            yprime_cross_z = yprime_cross_z / np.linalg.norm(yprime_cross_z)
            novelcam2fg[:3, 0] = yprime_cross_z

            # extracting Y-axis
            z_cross_x = np.cross(novelcam2fg[:3, 2], novelcam2fg[:3, 0])
            z_cross_x = z_cross_x / np.linalg.norm(z_cross_x)
            novelcam2fg[:3, 1] = z_cross_x

            '''
            asset_offsetabt_xaxis = np.radians(opts.asset_offsetabt_xaxis)
            asset_offsetabt_yaxis = np.radians(opts.asset_offsetabt_yaxis)
            asset_offsetabt_zaxis = np.radians(opts.asset_offsetabt_zaxis)
            rotation_abt_xaxis = np.array([[1, 0, 0, 0],
                                [0,  np.cos(asset_offsetabt_xaxis), np.sin(asset_offsetabt_xaxis), 0],
                                [0, -np.sin(asset_offsetabt_xaxis), np.cos(asset_offsetabt_xaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_yaxis = np.array([[np.cos(asset_offsetabt_yaxis), 0, -np.sin(asset_offsetabt_yaxis), 0],
                                [0, 1, 0, 0],
                                [np.sin(asset_offsetabt_yaxis), 0, np.cos(asset_offsetabt_yaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_zaxis = np.array([[np.cos(asset_offsetabt_zaxis), np.sin(asset_offsetabt_zaxis), 0, 0],
                                [-np.sin(asset_offsetabt_zaxis),  np.cos(asset_offsetabt_zaxis),    0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            '''
            firstpersoncam_offsetabt_xaxis = np.radians(opts.firstpersoncam_offsetabt_xaxis)
            firstpersoncam_offsetabt_yaxis = np.radians(opts.firstpersoncam_offsetabt_yaxis)
            firstpersoncam_offsetabt_zaxis = np.radians(opts.firstpersoncam_offsetabt_zaxis)
            rotation_abt_xaxis = np.array([[1, 0, 0, 0],
                                [0,  np.cos(firstpersoncam_offsetabt_xaxis), np.sin(firstpersoncam_offsetabt_xaxis), 0],
                                [0, -np.sin(firstpersoncam_offsetabt_xaxis), np.cos(firstpersoncam_offsetabt_xaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_yaxis = np.array([[np.cos(firstpersoncam_offsetabt_yaxis), 0, -np.sin(firstpersoncam_offsetabt_yaxis), 0],
                                [0, 1, 0, 0],
                                [np.sin(firstpersoncam_offsetabt_yaxis), 0, np.cos(firstpersoncam_offsetabt_yaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_zaxis = np.array([[np.cos(firstpersoncam_offsetabt_zaxis), np.sin(firstpersoncam_offsetabt_zaxis), 0, 0],
                                [-np.sin(firstpersoncam_offsetabt_zaxis),  np.cos(firstpersoncam_offsetabt_zaxis),    0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            novelcam2fg = np.matmul(novelcam2fg, rotation_abt_xaxis)
            novelcam2fg = np.matmul(novelcam2fg, rotation_abt_yaxis)
            novelcam2fg = np.matmul(novelcam2fg, rotation_abt_zaxis)            

            # translation component = coordinate of the designated vertex in the fgframe
            eps = opts.firstpersoncam_offset_z * opts_list[-1].dep_scale
            novelcam2fg[:3, 3] = mesh_fg_time.vertices[opts.fg_normalbase_vertex_index, :].copy() + mesh_fg_time.vertex_normals[opts.fg_normalbase_vertex_index, :].copy() * eps

            # TODO: shoudn't we be offseting along the z direction?
            novelcam2fg[:3, 3] = novelcam2fg[:3, 3] - opts.firstpersoncam_offset_y * opts_list[-1].dep_scale * novelcam2fg[:3, 1]
            novelcams2fg.append(novelcam2fg)
        
        novelcams2fg = np.stack(novelcams2fg, axis = 0)          # shape (N, 4, 4)
        
        novelcams2bkgd = np.matmul(fg2bkgd, novelcams2fg)
        bkgd2novelcams = np.linalg.inv(novelcams2bkgd)

        # extrinsics of video-view cameras (w.r.t. background)
        bkgd2videocams = np.copy(rtks_objs[-1])      # (N, 4, 4)
        bkgd2videocams[:, 3, :] = 0.
        bkgd2videocams[:, 3, 3] = 1.

        for rtks_obj in rtks_objs:
            # root-body pose for each object
            root2videocams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
            root2videocams[:,:3,:] = rtks_obj[:,:3,:]
            root2videocams[:,3,:] = 0.
            root2videocams[:,3,3] = 1.

            # root to novel-view transformation
            #TODO
            root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
            rtks_obj[:,:3,:] = root2novelcams[:,:3,:]
            
            # scaling camera intrinsics for wider fov
            rtks_obj[:,3] *= opts.scale_fps

    elif opts.thirdperson_view:
        opts.nvs_outpath = opts.nvs_outpath + "-tpsview"

        if opts.maxframe > 0:
            #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
            sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
        else:
            #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
            sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]

        # what should the novel-camera trajectory (w.r.t bkgd frame) be?
        # 
        # assuming that the fgmeshcenter frame is well aligned with the cat, we can set an offset along the xz plane of the fgmeshcenter frame and view the cat from there
        # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should a few units behind the cat in the z-direction
        bkgd2cam = np.concatenate((rtks_objs[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[-1].shape[0], axis = 0)), axis = 1)
        fg2cam = np.concatenate((rtks_objs[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
        cam2bkgd = np.linalg.inv(bkgd2cam)
        fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates  
        
        '''
        #mesh_rest_fg = trimesh.load('%s/mesh_rest-14.obj'%(rootdir_objs[0]),process=False)  # loading the mesh for the foreground
        mesh_rest_fg = trimesh.load('%s/mesh-rest.obj'%(rootdir_objs[0]),process=False)  # loading the mesh for the foreground
        mesh_rest_fg_center = mesh_rest_fg.vertices.mean(0)                                 # (3,)
        fgmeshcenter2fg = np.eye(4)
        fgmeshcenter2fg[:3, 3] = mesh_rest_fg_center                                        # coordinate of fgmeshcenter in fgroot frame
        thirdpersoncam2fgmeshcenter = np.eye(4)
        thirdpersoncam2fgmeshcenter[1, 3] = opts.thirdpersoncam_offset_y                    # offset along the y-direction of the fgmeshcenter frame for third-person view
        thirdpersoncam2fgmeshcenter[2, 3] = opts.thirdpersoncam_offset_z                    # offset along the z-direction of the fgmeshcenter frame for third-person view

        fgmeshcenter2bkgd = np.matmul(fg2bkgd, np.repeat(fgmeshcenter2fg[None, :, :], fg2bkgd.shape[0], axis = 0))
        thirdpersoncam2bkgd = np.matmul(fgmeshcenter2bkgd, np.repeat(thirdpersoncam2fgmeshcenter[None, :, :], fg2bkgd.shape[0], axis = 0))

        bkgd2novelcams = np.linalg.inv(thirdpersoncam2bkgd)
        # change direction of x-axis and y-axis (rows) to make them consistent with camera (NOT exactly sure why we have to do this but oh well)
        bkgd2novelcams[:, :2, :] *= -1
        '''

        meshdir_fg = opts_list[opts.fg_obj_index].rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
        novelcams2fg = []
        for i, frame_idx in enumerate(sample_idx):
            
            # loading the foreground mesh (choice of foreground number represented by opts.fg_obj_index) 
            meshdir_fg_time = glob.glob(meshdir_fg + "*-mesh-%05d.obj"%(frame_idx))
            assert(len(meshdir_fg_time) == 1)
            mesh_fg_time = trimesh.load(meshdir_fg_time[0], process=False)
            mest_fg_center_time = mesh_fg_time.vertices.mean(0)         # (3,)
            fgmeshcenter2fg_time = np.eye(4)
            fgmeshcenter2fg_time[:3, 3] = mest_fg_center_time                                       # coordinate of fgmeshcenter in fgroot frame
            tpscam2fgmeshcenter = np.eye(4)

            # translation component = coordinate of the designated vertex in the fgframe
            tpscam2fgmeshcenter[0, 3] = opts.thirdpersoncam_offset_x * opts_list[-1].dep_scale                    # offset along the x-direction of the fgmeshcenter frame for third-person view
            tpscam2fgmeshcenter[1, 3] = (opts.thirdpersoncam_fgmeshcenter_elevate_y + opts.thirdpersoncam_offset_y) * opts_list[-1].dep_scale                    # offset along the y-direction of the fgmeshcenter frame for third-person view
            tpscam2fgmeshcenter[2, 3] = opts.thirdpersoncam_offset_z * opts_list[-1].dep_scale                    # offset along the z-direction of the fgmeshcenter frame for third-person view
            
            # rotation component
            # extracting Z-axis direction of 3rd-person camera frame w.r.t. fgcenter frame = (-offset_x, -offset_y, -offset_z)
            view_dir = np.array([-opts.thirdpersoncam_offset_x * opts_list[-1].dep_scale, -opts.thirdpersoncam_offset_y * opts_list[-1].dep_scale, -opts.thirdpersoncam_offset_z * opts_list[-1].dep_scale])
            view_dir = view_dir / np.linalg.norm(view_dir)
            tpscam2fgmeshcenter[:3, 2] = view_dir

            # extracting X-axis direction of 3rd-person camera frame w.r.t. fgcenter frame 
            # cross product of z-axis and downward vector (y-prime) i.e. (0, -offset_y, 0)
            y_prime = np.array([0, -opts.thirdpersoncam_offset_y * opts_list[-1].dep_scale, 0])
            yprime_cross_z = np.cross(y_prime, tpscam2fgmeshcenter[:3, 2])
            yprime_cross_z = yprime_cross_z / np.linalg.norm(yprime_cross_z)
            tpscam2fgmeshcenter[:3, 0] = yprime_cross_z

            # extracting Y-axis
            z_cross_x = np.cross(tpscam2fgmeshcenter[:3, 2], tpscam2fgmeshcenter[:3, 0])
            z_cross_x = z_cross_x / np.linalg.norm(z_cross_x)
            tpscam2fgmeshcenter[:3, 1] = z_cross_x

            '''
            asset_offsetabt_xaxis = np.radians(opts.asset_offsetabt_xaxis)
            asset_offsetabt_yaxis = np.radians(opts.asset_offsetabt_yaxis)
            asset_offsetabt_zaxis = np.radians(opts.asset_offsetabt_zaxis)
            rotation_abt_xaxis = np.array([[1, 0, 0, 0],
                                [0,  np.cos(asset_offsetabt_xaxis), np.sin(asset_offsetabt_xaxis), 0],
                                [0, -np.sin(asset_offsetabt_xaxis), np.cos(asset_offsetabt_xaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_yaxis = np.array([[np.cos(asset_offsetabt_yaxis), 0, -np.sin(asset_offsetabt_yaxis), 0],
                                [0, 1, 0, 0],
                                [np.sin(asset_offsetabt_yaxis), 0, np.cos(asset_offsetabt_yaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_zaxis = np.array([[np.cos(asset_offsetabt_zaxis), np.sin(asset_offsetabt_zaxis), 0, 0],
                                [-np.sin(asset_offsetabt_zaxis),  np.cos(asset_offsetabt_zaxis),    0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            '''
            thirdpersoncam_offsetabt_xaxis = np.radians(opts.thirdpersoncam_offsetabt_xaxis)
            thirdpersoncam_offsetabt_yaxis = np.radians(opts.thirdpersoncam_offsetabt_yaxis)
            thirdpersoncam_offsetabt_zaxis = np.radians(opts.thirdpersoncam_offsetabt_zaxis)
            rotation_abt_xaxis = np.array([[1, 0, 0, 0],
                                [0,  np.cos(thirdpersoncam_offsetabt_xaxis), np.sin(thirdpersoncam_offsetabt_xaxis), 0],
                                [0, -np.sin(thirdpersoncam_offsetabt_xaxis), np.cos(thirdpersoncam_offsetabt_xaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_yaxis = np.array([[np.cos(thirdpersoncam_offsetabt_yaxis), 0, -np.sin(thirdpersoncam_offsetabt_yaxis), 0],
                                [0, 1, 0, 0],
                                [np.sin(thirdpersoncam_offsetabt_yaxis), 0, np.cos(thirdpersoncam_offsetabt_yaxis), 0],
                                [0, 0, 0, 1]])
            rotation_abt_zaxis = np.array([[np.cos(thirdpersoncam_offsetabt_zaxis), np.sin(thirdpersoncam_offsetabt_zaxis), 0, 0],
                                [-np.sin(thirdpersoncam_offsetabt_zaxis),  np.cos(thirdpersoncam_offsetabt_zaxis),    0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            tpscam2fgmeshcenter = np.matmul(tpscam2fgmeshcenter, rotation_abt_xaxis)
            tpscam2fgmeshcenter = np.matmul(tpscam2fgmeshcenter, rotation_abt_yaxis)
            tpscam2fgmeshcenter = np.matmul(tpscam2fgmeshcenter, rotation_abt_zaxis)   

            novelcam2fg = np.matmul(tpscam2fgmeshcenter, fgmeshcenter2fg_time)
            novelcams2fg.append(novelcam2fg)
        
        novelcams2fg = np.stack(novelcams2fg, axis = 0)          # shape (N, 4, 4)
        novelcams2bkgd = np.matmul(fg2bkgd, novelcams2fg)
        bkgd2novelcams = np.linalg.inv(novelcams2bkgd)

        # extrinsics of video-view cameras (w.r.t. background)
        bkgd2videocams = np.copy(rtks_objs[-1])      # (N, 4, 4)
        bkgd2videocams[:, 3, :] = 0.
        bkgd2videocams[:, 3, 3] = 1.

        for rtks_obj in rtks_objs:
            # root-body pose for each object
            root2videocams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
            root2videocams[:,:3,:] = rtks_obj[:,:3,:]
            root2videocams[:,3,:] = 0.
            root2videocams[:,3,3] = 1.

            # root to novel-view transformation
            #TODO
            root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
            rtks_obj[:,:3,:] = root2novelcams[:,:3,:]

            # scaling camera intrinsics for wider fov
            rtks_obj[:,3] *= opts.scale_tps

    # novel-view synthesis from fixed view
    elif opts.fixed_view or opts.topdown_view:
        if opts.maxframe > 0:
            sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
        else:
            sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)

        if opts.fix_frame >= 0:
            # camera view fixed to a specific frame's view
            rtks_objs_fixframe = [rtks[np.repeat(opts.fix_frame, size)] for rtks in rtks_objs]
            rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]
            
            # topdown view corresponding to the camera view of "fix_frame"
            if opts.topdown_view:
                opts.nvs_outpath = opts.nvs_outpath + "-bev"

                # assuming that the fgmeshcenter frame is well aligned with the fg object itself, we can define an offset along the y axis of the root frame of the fg and view the fg object from there
                # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should be a few units in the y-direction
                bkgd2cam = np.concatenate((rtks_objs_fixframe[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fixframe[-1].shape[0], axis = 0)), axis = 1)
                fg2cam = np.concatenate((rtks_objs_fixframe[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fixframe[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
                cam2bkgd = np.linalg.inv(bkgd2cam)
                fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates                  
                
                topdowncam2fg = np.eye(4)
                topdowncam2fg[:3, :3] = cv2.Rodrigues(np.asarray([np.pi/2., 0., 0.]))[0]
                
                topdowncam_offsetabt_xaxis = np.radians(opts.topdowncam_offsetabt_xaxis)
                topdowncam_offsetabt_yaxis = np.radians(opts.topdowncam_offsetabt_yaxis)
                topdowncam_offsetabt_zaxis = np.radians(opts.topdowncam_offsetabt_zaxis)
                rotation_abt_xaxis = np.array([[1, 0, 0, 0],
                                    [0,  np.cos(topdowncam_offsetabt_xaxis), np.sin(topdowncam_offsetabt_xaxis), 0],
                                    [0, -np.sin(topdowncam_offsetabt_xaxis), np.cos(topdowncam_offsetabt_xaxis), 0],
                                    [0, 0, 0, 1]])
                rotation_abt_yaxis = np.array([[np.cos(topdowncam_offsetabt_yaxis), 0, -np.sin(topdowncam_offsetabt_yaxis), 0],
                                    [0, 1, 0, 0],
                                    [np.sin(topdowncam_offsetabt_yaxis), 0, np.cos(topdowncam_offsetabt_yaxis), 0],
                                    [0, 0, 0, 1]])
                rotation_abt_zaxis = np.array([[np.cos(topdowncam_offsetabt_zaxis), np.sin(topdowncam_offsetabt_zaxis), 0, 0],
                                    [-np.sin(topdowncam_offsetabt_zaxis),  np.cos(topdowncam_offsetabt_zaxis),    0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
                topdowncam2fg = np.matmul(topdowncam2fg, rotation_abt_xaxis)
                topdowncam2fg = np.matmul(topdowncam2fg, rotation_abt_yaxis)
                topdowncam2fg = np.matmul(topdowncam2fg, rotation_abt_zaxis)

                topdowncam2fg[0, 3] = opts.topdowncam_offset_x                                                # offset along the x-direction of the fgroot frame for topdown view
                topdowncam2fg[1, 3] = opts.topdowncam_offset_y                                                # offset along the y-direction of the fgroot frame for topdown view
                topdowncam2fg[2, 3] = opts.topdowncam_offset_z
                
                elevatedfg2bkgd = np.matmul(fg2bkgd, np.repeat(topdowncam2fg[None, ...], fg2bkgd.shape[0], axis = 0))
                bkgd2novelcams = np.linalg.inv(elevatedfg2bkgd)
                #bkgd2novelcams[:, :2, :] *= -1
            
            else:
                opts.nvs_outpath = opts.nvs_outpath + "-fixedview"
                bkgd2novelcams = np.copy(rtks_objs_fixframe[-1])

            # extrinsics of video-view cameras (w.r.t. background)
            bkgd2videocams = np.copy(rtks_objs[-1])      # (N, 4, 4)
            bkgd2videocams[:, 3, :] = 0.
            bkgd2videocams[:, 3, 3] = 1.

            for rtks_obj in rtks_objs:
                # root-body pose for each object
                root2videocams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
                root2videocams[:,:3,:] = rtks_obj[:,:3,:]
                root2videocams[:,3,:] = 0.
                root2videocams[:,3,3] = 1.

                # root to novel-view transformation
                #TODO
                root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
                rtks_obj[:,:3,:] = root2novelcams[:,:3,:]

        # camera view fixed to user-defined camera view 
        else:
            pass
    
    # novel-view synthesis from 2nd camera of dual-camera rig
    elif opts.stereo_view:

        opts.nvs_outpath = opts.nvs_outpath + "-stereoview"

        if opts.maxframe > 0:
            sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
        else:
            sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]

        # compute the relative transformation from the reference camera to the 2nd camera of the dualrig 
        # TO DO: refcam is also used in fixed-view synthesis - fix this redundancy
        refcam = opts.refcam                    # refcam = "leftcam" OR "rightcam"
        if refcam == "leftcam":
            secondcam = "rightcam"
        if refcam == "rightcam":
            secondcam = "leftcam"
        
        config = configparser.RawConfigParser()
        seqname_secondcam = opts.seqname.replace(refcam, secondcam)
        config.read('configs/%s.config'%seqname_secondcam)
        ks_secondcam = np.array(config.get('data_0', 'ks').split(" ")).astype(float)
        K_secondcam = np.array([[ks_secondcam[0], 0., ks_secondcam[2]],
                                [0., ks_secondcam[1], ks_secondcam[3]],
                                [0., 0., 1.]])

        gt_imglist_secondcam = [gt_imgpath.replace(refcam, secondcam) for gt_imgpath in gt_imglist_refcam]
        gt_deplist_secondcam = [gt_deppath.replace(refcam, secondcam) for gt_deppath in gt_deplist_refcam]
        gt_conflist_secondcam = [gt_confpath.replace(refcam, secondcam) for gt_confpath in gt_conflist_refcam]

        # multi-obj case
        #gt_masklist_secondcam = [gt_maskpath.replace(refcam, secondcam) for gt_maskpath in gt_masklist_refcam]
        gt_masklist_secondcam_objs = [[gt_maskpath.replace(refcam, secondcam) for gt_maskpath in gt_masklist_refcam] for gt_masklist_refcam in gt_masklist_refcam_objs]

        # for correspondence computation using optical flow
        from models.VCNplus import VCN
        from models.VCNplus import WarpModule, flow_reg
        flow_model = VCN([1, 256, 256], md=[int(4*(opts.maxdisp/256)),4,4,4,4], fac=opts.fac)
        flow_model = torch.nn.DataParallel(flow_model)
        flow_model.cuda()

        if opts.loadmodel is not None:
            pretrained_dict = torch.load(opts.loadmodel)
            mean_L=pretrained_dict['mean_L']
            mean_R=pretrained_dict['mean_R']
            pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
            flow_model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        else:
            print('dry run')

        if opts.flow_correspondence:
            refcam2secondcams = []
            P_secondcam_valids = []
            p_refcam_valids = []
            for i, frame_idx in enumerate(sample_idx):
                # a) compute correspondences between stereo image pairs and their confidence scores using optical flow
                
                rgb_gt_refcam = cv2.imread(gt_imglist_refcam[frame_idx])[:,:,::-1]
                # TODO: change "gt_masklist_refcam" to multi-obj case
                mask_gt_refcam = cv2.imread(gt_masklist_refcam[frame_idx], 0)
                rgb_gt_secondcam = cv2.imread(gt_imglist_secondcam[frame_idx])[:,:,::-1]
                # TODO: change "gt_masklist_refcam" to multi-obj case
                mask_gt_secondcam = cv2.imread(gt_masklist_secondcam[frame_idx], 0)

                mask_gt_refcam = mask_gt_refcam / np.sort(np.unique(mask_gt_refcam))[1]
                occluder_refcam = mask_gt_refcam==255
                mask_gt_refcam[occluder_refcam] = 0
                mask_gt_refcam = np.logical_and(mask_gt_refcam>0, mask_gt_refcam!=255)

                mask_gt_secondcam = mask_gt_secondcam / np.sort(np.unique(mask_gt_secondcam))[1]
                occluder_secondcam = mask_gt_secondcam==255
                mask_gt_secondcam[occluder_secondcam] = 0
                mask_gt_secondcam = np.logical_and(mask_gt_secondcam>0, mask_gt_secondcam!=255)

                dph_gt_refcam = read_depth(gt_deplist_refcam[frame_idx])
                conf_gt_refcam = read_conf(gt_conflist_refcam[frame_idx])
                conf_gt_refcam[np.isnan(dph_gt_refcam)] = 0.
                dph_gt_refcam[np.isnan(dph_gt_refcam)] = 4.
                ################## ignoring pixels whose depth values are close to camera ############
                conf_gt_refcam[dph_gt_refcam > 2.0] = 0.
                ######################################################################################

                dph_gt_refcam = dph_gt_refcam * opts_list[-1].dep_scale                 # max depth recorded by the LiDAR is 4m 
                dph_gt_refcam = cv2.resize(dph_gt_refcam, rndsils.shape[1:][::-1], interpolation=cv2.INTER_LINEAR)
                conf_gt_refcam = cv2.resize(conf_gt_refcam, rndsils.shape[1:][::-1], interpolation=cv2.INTER_NEAREST)

                dph_gt_secondcam = read_depth(gt_deplist_secondcam[frame_idx])
                conf_gt_secondcam = read_conf(gt_conflist_secondcam[frame_idx])
                conf_gt_secondcam[np.isnan(dph_gt_secondcam)] = 0.
                dph_gt_secondcam[np.isnan(dph_gt_secondcam)] = 4.
                ################## ignoring pixels whose depth values are close to camera ############
                conf_gt_secondcam[dph_gt_secondcam > 2.0] = 0.
                ######################################################################################

                dph_gt_secondcam = dph_gt_secondcam * opts_list[-1].dep_scale           # max depth recorded by the LiDAR is 4m
                dph_gt_secondcam = cv2.resize(dph_gt_secondcam, rndsils.shape[1:][::-1], interpolation=cv2.INTER_LINEAR)
                conf_gt_secondcam = cv2.resize(conf_gt_secondcam, rndsils.shape[1:][::-1], interpolation=cv2.INTER_NEAREST)

                # flowfw.shape = (H, W, 3)
                # flowfw[..., 0]: x-displacement (right direction is +ve)
                # flowfw[..., 1]: y-displacement (left direction is +ve)
                # occfw.shape = (H, W)
                flowfw, occfw = flow_inference(rgb_gt_refcam, rgb_gt_secondcam, flow_model, mean_L, mean_R, opts)        
                flowvis = flowfw.copy() * (occfw[..., np.newaxis] > 0)

                #flowvis[~mask_gt_refcam]=0     # comment for easier visualization
                flowvis_overlay_rgbgtrefcam = point_vec(rgb_gt_refcam, flowvis)
                flowvis_overlay_rgbgtsecondcam = point_vec(rgb_gt_secondcam, flowvis)
                flowvis_overlay = np.concatenate([flowvis_overlay_rgbgtrefcam, flowvis_overlay_rgbgtsecondcam], axis = 1)
                cv2.imwrite('%s-flowvis_%05d.png'%(opts.nvs_outpath, frame_idx), flowvis_overlay)
                
                # b) filter correspondences using flow confidence scores, GT confmaps from the second camera, and the image bounds
                rgb_height = rgb_gt_refcam.shape[0]
                rgb_width = rgb_gt_refcam.shape[1]
                x_coord_refcam, y_coord_refcam = np.meshgrid(np.arange(rgb_gt_refcam.shape[1]), np.arange(rgb_gt_refcam.shape[0]))              # shape = (H, W)
                
                p_refcam = np.stack([x_coord_refcam, y_coord_refcam], axis = -1)                                                                # shape = (H, W, 2)
                flow_xy = flowfw[..., :2]                                                                                                       # shape = (H, W, 2)
                p_secondcam = p_refcam + flow_xy                                                                                                # shape = (H, W, 2)
                
                p_refcam = p_refcam.reshape(-1, 2)                                                                                              # shape = (N = H*W, 2)
                p_secondcam = p_secondcam.reshape(-1, 2)                                                                                        # shape = (N = H*W, 2)
                
                pixels_within_bound = (p_secondcam[..., 0] >= 0) & (p_secondcam[..., 0] <= rgb_width - 1) & (p_secondcam[..., 1] >= 0) & (p_secondcam[..., 1] <= rgb_height - 1)            
                x_coord_secondcam = p_secondcam[:, 0]                                                                                           # shape = (N)
                y_coord_secondcam = p_secondcam[:, 1]                                                                                           # shape = (N)
                x_coord_secondcam_clip = np.clip(x_coord_secondcam, 0, rgb_width - 1)
                y_coord_secondcam_clip = np.clip(y_coord_secondcam, 0, rgb_height - 1)

                conf_secondcam = conf_gt_secondcam[np.round(y_coord_secondcam_clip).astype(int), np.round(x_coord_secondcam_clip).astype(int)]  # shape = (N = H*W)                                                     # shape = (N)

                ################## ignoring pixels whose depth values are close to camera ############
                conf_refcam = conf_gt_refcam[p_refcam[:, 1].astype(int), p_refcam[:, 0].astype(int)]                                                                # shape = (N = H*W)
                pixels_with_highconf = (conf_secondcam >= 1.5) * (conf_refcam >= 1.5)
                ################## ignoring pixels whose depth values are close to camera ############

                pixels_with_certainflow = occfw.reshape(-1) > 0                                                                                                     # shape = (N)
                pixels_valid = pixels_within_bound & pixels_with_highconf & pixels_with_certainflow                                                                 # shape = (N)

                x_coord_refcam_valid = p_refcam[:, 0][pixels_valid]
                y_coord_refcam_valid = p_refcam[:, 1][pixels_valid]
                p_refcam_valid = np.stack([x_coord_refcam_valid, y_coord_refcam_valid], axis = -1)                                                                  # shape = (N_valid, 2)

                focal_refcam = rtks_objs[-1][i, 3, :2]
                ppoint_refcam = rtks_objs[-1][i, 3, 2:]
                K_refcam = np.array([[focal_refcam[0], 0., ppoint_refcam[0]],                                                                            
                                    [0., focal_refcam[1], ppoint_refcam[1]],
                                    [0., 0., 1.]])                                                                                                                  # shape = (3, 3)
                
                x_coord_secondcam_valid = x_coord_secondcam[pixels_valid]                                                                                           # shape = (N_valid)
                y_coord_secondcam_valid = y_coord_secondcam[pixels_valid]                                                                                           # shape = (N_valid)
                p_homo_secondcam_valid = np.stack([x_coord_secondcam_valid, y_coord_secondcam_valid, np.ones_like(x_coord_secondcam_valid)], axis=-1)               # shape = (N_valid, 3)
                depth_secondcam_valid = dph_gt_secondcam.reshape(-1)[pixels_valid]                                                                                  # shape = (N_valid)
                P_secondcam_valid = np.repeat(depth_secondcam_valid[:, np.newaxis], 3, axis=-1) * np.matmul(p_homo_secondcam_valid, np.linalg.inv(K_secondcam.T))   # shape = (N_valid, 3)

                # c) for each camera, compute the transformation from the other camera to the given camera
                #    by setting the coordinate system of the given camera as the world frame
                #for camera_id, camera_name in camera_names:
                    # using GT depthmap, GT confmap, and (potentially optimized) camera intrinsics of {camera_name}
                    # compute 3d coordinates of the corresponding points in world frame:
                    #
                    # P = z * inv(K) * \tilde{p} (P = 3x1 3d coordinates, z = depth, K = 3x3 camera intrinsics, and \tilde{p} = 3x1 homogeneous coordinates of 2d point p)

                    # call cv2.solvePnP to compute the relative transformation from the other camera to the given camera    
                    
                    # from othercam2givencam compute refcam2secondcam
                    #refcam2secondcam = NotImplemented          # T_{2ndcam} = T_{rel} x T_{refcam}
                                                                # T_{refcam} = bkgd2refcams
                                                                # T_{2ndcam} = bkgd2secondcams
                
                """
                _, rvec_init, tvec_init, inliers_init = cv2.solvePnPRansac(P_secondcam_valid.astype('float'), 
                                                    p_refcam_valid.astype('float'), 
                                                    K_refcam, 
                                                    0,
                                                    reprojectionError = 20.0,
                                                    flags=cv2.SOLVEPNP_DLS)

                _, rvec_finetuned, tvec_finetuned, inliers_finetuned = cv2.solvePnPRansac(P_secondcam_valid.astype('float'),
                                                                p_refcam_valid.astype('float'),
                                                                K_refcam,
                                                                0,
                                                                rvec_init,
                                                                tvec_init,
                                                                useExtrinsicGuess=True,
                                                                reprojectionError = 20.0,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

                print("inliers_init: {}".format(len(inliers_init) / (rgb_width * rgb_height)))                  # returns True
                print("inliers_finetuned: {}".format(len(inliers_finetuned) / (rgb_width * rgb_height)))        # returns True
                #print("retval_init: {}".format(retval_init))                                                   # returns True
                #print("retval_finetuned: {}".format(retval_finetuned))                                         # returns True
                """

                P_secondcam_valids.append(P_secondcam_valid)
                p_refcam_valids.append(p_refcam_valid)

                '''
                _, rvec_init, tvec_init = cv2.solvePnP(P_secondcam_valid[..., np.newaxis].astype('float'), 
                                                    p_refcam_valid[..., np.newaxis].astype('float'), 
                                                    K_refcam, 
                                                    0,
                                                    flags=cv2.SOLVEPNP_DLS)

                _, rvec_finetuned, tvec_finetuned = cv2.solvePnP(P_secondcam_valid[..., np.newaxis].astype('float'),
                                                                p_refcam_valid[..., np.newaxis].astype('float'),
                                                                K_refcam,
                                                                0,
                                                                rvec_init,
                                                                tvec_init,
                                                                useExtrinsicGuess=True,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

                # compute reprojection error
                reprojected_p_refcam, _, = cv2.projectPoints(P_secondcam_valid[..., np.newaxis].astype('float'),
                                                            rvec_finetuned,
                                                            tvec_finetuned,
                                                            K_refcam,
                                                            0)
                
                reprojected_p_refcam = reprojected_p_refcam[:, 0, :]    # reprojected_p_refcam.shape = (N, 1, 2)
                reprojected_vs_originalimgpoints = np.concatenate([p_refcam_valid, reprojected_p_refcam], axis = -1)        # shape = (N, 4)

                #print("reprojected vs original img points: {}".format(reprojected_vs_originalimgpoints))
                reproj_err = np.mean(np.linalg.norm(reprojected_p_refcam - p_refcam_valid, axis = -1))
                print("averaged reprojection error [units: pixels]: {}".format(reproj_err))

                secondcam2refcam_rot = cv2.Rodrigues(rvec_finetuned)[0]
                secondcam2refcam = np.eye(4)
                secondcam2refcam[:3, :3] = secondcam2refcam_rot                                     # shape = (3, 3)
                secondcam2refcam[:3, 3:4] = tvec_finetuned                                          # shape = (3, 1)
                refcam2secondcam = np.linalg.inv(secondcam2refcam)
                
                if reproj_err < 50:
                    print("refcam2secondcam: {}".format(refcam2secondcam))
                refcam2secondcams.append(refcam2secondcam)

            # average the refcam2second transformation matrices across camera_ids and frames
            refcam2secondcams = np.stack(refcam2secondcams, axis = 0)                               # shape = (N, 4, 4)
            refcam2secondcam = np.mean(refcam2secondcams, axis = 0)                                 # shape = (4, 4)
            '''

            P_secondcam_valids = np.concatenate(P_secondcam_valids, axis = 0)
            p_refcam_valids = np.concatenate(p_refcam_valids, axis = 0)
            
            """
            _, rvec_init, tvec_init = cv2.solvePnP(P_secondcam_valids[..., np.newaxis].astype('float'), 
                                                    p_refcam_valids[..., np.newaxis].astype('float'), 
                                                    K_refcam, 
                                                    0,
                                                    flags=cv2.SOLVEPNP_DLS)

            _, rvec_finetuned, tvec_finetuned = cv2.solvePnP(P_secondcam_valids[..., np.newaxis].astype('float'),
                                                            p_refcam_valids[..., np.newaxis].astype('float'),
                                                            K_refcam,
                                                            0,
                                                            rvec_init,
                                                            tvec_init,
                                                            useExtrinsicGuess=True,
                                                            flags=cv2.SOLVEPNP_ITERATIVE)
            """
            _, rvec_init, tvec_init, inliers_init = cv2.solvePnPRansac(P_secondcam_valids.astype('float'), 
                                    p_refcam_valids.astype('float'), 
                                    K_refcam, 
                                    0,
                                    iterationsCount = 100,
                                    reprojectionError = 5.0,
                                    flags=cv2.SOLVEPNP_DLS)

            _, rvec_finetuned, tvec_finetuned, inliers_finetuned = cv2.solvePnPRansac(P_secondcam_valids.astype('float'),
                                                            p_refcam_valids.astype('float'),
                                                            K_refcam,
                                                            0,
                                                            rvec_init,
                                                            tvec_init,
                                                            useExtrinsicGuess=True,
                                                            iterationsCount = 100,
                                                            reprojectionError = 5.0,
                                                            flags=cv2.SOLVEPNP_ITERATIVE)

            print("inliers_init: {}".format(len(inliers_init) / (rgb_width * rgb_height)))                  # returns True
            print("inliers_finetuned: {}".format(len(inliers_finetuned) / (rgb_width * rgb_height)))        # returns True
            P_secondcam_valids = P_secondcam_valids[inliers_finetuned[:, 0], :]
            p_refcam_valids = p_refcam_valids[inliers_finetuned[:, 0], :]

            # compute reprojection error
            reprojected_p_refcam, _, = cv2.projectPoints(P_secondcam_valids[..., np.newaxis].astype('float'),
                                                        rvec_finetuned,
                                                        tvec_finetuned,
                                                        K_refcam,
                                                        0)
            
            reprojected_p_refcam = reprojected_p_refcam[:, 0, :]    # reprojected_p_refcam.shape = (N, 1, 2)
            reproj_err = np.mean(np.linalg.norm(reprojected_p_refcam - p_refcam_valids, axis = -1))
            print("averaged reprojection error [units: pixels]: {}".format(reproj_err))

            secondcam2refcam_rot = cv2.Rodrigues(rvec_finetuned)[0]
            secondcam2refcam = np.eye(4)
            secondcam2refcam[:3, :3] = secondcam2refcam_rot                                     # shape = (3, 3)
            secondcam2refcam[:3, 3:4] = tvec_finetuned                                          # shape = (3, 1)
            refcam2secondcam = np.linalg.inv(secondcam2refcam)

        # ref2cam2second transformation computed from manually specified correspondences
        else:
            # assume normrefcam2secondcam.npy is the relative transformation defined in the metric space
            refcam2secondcam = np.load("logdir/{}/normrefcam2secondcam.npy".format(opts.seqname))       # shape = (4, 4)
            
            # need to apply scaling to the translations for proper novel-view synthesis
            refcam2secondcam[:3, 3:4] *= opts_list[-1].dep_scale

        # 2. compute bkgd to novel-view transformation
        for rtks_obj in rtks_objs:
            # root-body pose for each object
            root2videocams = np.tile(np.eye(4)[None, ...], (size, 1, 1))
            root2videocams[:,:3,:] = rtks_obj[:,:3,:]
            root2videocams[:,3,:] = 0.
            root2videocams[:,3,3] = 1.

            # root to novel-view transformation
            #root2novelcams = np.matmul(np.matmul(bkgd2novelcams, np.linalg.inv(bkgd2videocams)), root2videocams)
            root2novelcams = np.matmul(refcam2secondcam, root2videocams)
            rtks_obj[:,:3,:] = root2novelcams[:,:3,:]
        
    # reconstruction from input-video view
    #else:
    elif opts.input_view:
        opts.nvs_outpath = opts.nvs_outpath + "-inputview"

        if opts.maxframe > 0:
            #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
            sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
        else:
            #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
            sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
        rtks_objs = [rtks[sample_idx, ...] for rtks in rtks_objs]       # a list of [4, 4] camera matrices

    ####################################################################################################
    ####################################################################################################
    # determine render image scale
    #rtks[:,3] = rtks[:,3]*opts.scale                # scaling intrinsics
    #bs = len(rtks)
    for rtks in rtks_objs:
        rtks[:,3] = rtks[:,3]*opts.scale             # scaling intrinsics

    if opts.render_cam:
        for rtks in rtks_objs_rendercam:
            rtks[:,3] = rtks[:,3]*opts.scale        # scaling intrinsics
    
    if opts.stereo_view:
        K_secondcam[:2,:3] = K_secondcam[:2,:3] * opts.scale

    bs = len(rtks_objs[-1])
    img_size = int(max(img_size)*opts.scale)
    print("render size: %d"%img_size)
    model.img_size = img_size
    opts.render_size = img_size

    if img_type=='vert':
        size_short_edge = int(rndsils[0].shape[1] * img_size/rndsils[0].shape[0])
        height = img_size
        width = size_short_edge        
    else:
        size_short_edge = int(rndsils[0].shape[0] * img_size/rndsils[0].shape[1])
        height = size_short_edge
        width = img_size

    # mesh renderings
    # prerequisites:
    #
    # - rtks_objs: a list of obj2novelcam transformations of shape (N, 4, 4)
    # - meshes_objs: a dict where each entry contains a list of N meshes of each object (fg / bkgd)
    #
    direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
    rtks_objs_torch = [torch.Tensor(rtks_obj).cuda() for rtks_obj in rtks_objs]
    if opts.render_cam:
        rtks_objs_rendercam_torch = [torch.Tensor(rtks_obj_rendercam).cuda() for rtks_obj_rendercam in rtks_objs_rendercam]

    if opts.filter_3d or opts.render_cam:
        print("RENDERING 3D ASSET")
        # load asset mesh
        if opts.filter_3d:
            resolver = trimesh.resolvers.FilePathResolver('mesh_material/UnicornHorn_OBJ/')
            mesh_asset = trimesh.load(file_obj='mesh_material/UnicornHorn_OBJ/UnicornHorn.obj', resolver = resolver, process=False)
            color = mesh_asset.visual.to_color()
            mesh_asset.visual.vertex_colors = color
            mesh_asset.visual.vertex_colors.vertex_colors = np.tile(np.array([[0.439216, 0.9137255, 0.99607843, 1.]]), (mesh_asset.visual.vertex_colors.vertex_colors.shape[0], 1))

            # rotate the unicorn horn -90 degrees abt x-axis to make unicorn horn compliant with OpenGL convention (where the normal / principal viewing direction points towards -ve z-axis)
            rotation_abt_xaxis = np.array([[1, 0, 0],
                               [0,  np.cos(np.pi / 2), np.sin(np.pi / 2)],
                               [0, -np.sin(np.pi / 2), np.cos(np.pi / 2)]])
            mesh_asset.vertices = np.matmul(rotation_abt_xaxis, mesh_asset.vertices.T).T

        if opts.render_cam:
            mesh_asset = trimesh.load(file_obj='mesh_material/camera.obj', force='mesh')
            color = mesh_asset.visual.to_color()
            mesh_asset.visual.vertex_colors = color
            #mesh_asset.visual.vertex_colors.vertex_colors = np.tile(np.array([[0.1, 0.1, 0.1, 1.]]), (mesh_asset.visual.vertex_colors.vertex_colors.shape[0], 1))
            if opts.thirdperson_view:
                # color blue
                mesh_asset.visual.vertex_colors.vertex_colors = np.tile(np.array([[0, 0, 0.5, 1.]]), (mesh_asset.visual.vertex_colors.vertex_colors.shape[0], 1))
            elif opts.firstperson_view:
                # color yellow
                mesh_asset.visual.vertex_colors.vertex_colors = np.tile(np.array([[0.5, 0.5, 0, 1.]]), (mesh_asset.visual.vertex_colors.vertex_colors.shape[0], 1))

        '''
        asset_offsetabt_xaxis = np.radians(opts.asset_offsetabt_xaxis)
        asset_offsetabt_yaxis = np.radians(opts.asset_offsetabt_yaxis)
        asset_offsetabt_zaxis = np.radians(opts.asset_offsetabt_zaxis)
        rotation_abt_xaxis = np.array([[1, 0, 0],
                               [0,  np.cos(asset_offsetabt_xaxis), np.sin(asset_offsetabt_xaxis)],
                               [0, -np.sin(asset_offsetabt_xaxis), np.cos(asset_offsetabt_xaxis)]])
        rotation_abt_yaxis = np.array([[np.cos(asset_offsetabt_yaxis), 0, -np.sin(asset_offsetabt_yaxis)],
                               [0, 1, 0],
                               [np.sin(asset_offsetabt_yaxis), 0, np.cos(asset_offsetabt_yaxis)]])
        rotation_abt_zaxis = np.array([[np.cos(asset_offsetabt_zaxis), np.sin(asset_offsetabt_zaxis), 0],
                               [-np.sin(asset_offsetabt_zaxis),  np.cos(asset_offsetabt_zaxis),    0],
                               [0, 0, 1]])
        mesh_asset.vertices = np.matmul(rotation_abt_xaxis, mesh_asset.vertices.T).T
        mesh_asset.vertices = np.matmul(rotation_abt_yaxis, mesh_asset.vertices.T).T
        mesh_asset.vertices = np.matmul(rotation_abt_zaxis, mesh_asset.vertices.T).T
        '''

        # scale vertices of 3d asset
        mesh_asset.vertices = mesh_asset.vertices * opts.asset_scale

        meshasset_rnd_colors = []
        meshasset_rnd_depths = []

    print("RENDERING MESH")
    mesh_rnd_colors = []
    near_far_values = []
    
    bkgd_meshes= []  
    bkgd_mesh_rnd_depths = []      

    for i, frame_idx in enumerate(sample_idx):    
        print("index {} / {}".format(i, len(sample_idx)))
        r = OffscreenRenderer(img_size, img_size)
        scene = Scene(ambient_light=0.6*np.asarray([1.,1.,1.,1.]))
    
        for obj_index, opts_obj in enumerate(opts_list):
            meshdir_obj = opts_obj.rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
            rtks_obj = torch.Tensor(rtks_objs[obj_index]).cuda()
        
            # 1. load meshes
            meshdir_obj_time = glob.glob(meshdir_obj + "*-mesh-%05d.obj"%(frame_idx))
            assert(len(meshdir_obj_time) == 1)
            mesh_obj_time = trimesh.load(meshdir_obj_time[0], process=False)

            mesh_obj_time.visual.vertex_colors[:, :3] = 64
            if obj_index < len(opts_list) - 1:
                mesh_obj_time.visual.vertex_colors[:, obj_index] = 160     # obj0 (foreground): red, # obj 1 (background): green

            # 2. using rtks_objs, which represent obj2cam transformations, convert mesh vertices coords into the camera-space
            faces_obj_time = torch.Tensor(mesh_obj_time.faces[None]).cuda()
            verts_obj_time = torch.Tensor(mesh_obj_time.vertices[None]).cuda()
            
            if opts.render_cam:
                Rmat_obj_time = rtks_objs_rendercam_torch[obj_index][i : i+1, :3, :3]
                Tmat_obj_time = rtks_objs_rendercam_torch[obj_index][i : i+1, :3, 3]
                #Rmat_obj_time = rtks_objs_torch[obj_index][i : i+1, :3, :3]
                #Tmat_obj_time = rtks_objs_torch[obj_index][i : i+1, :3, 3]
            else:
                Rmat_obj_time = rtks_objs_torch[obj_index][i : i+1, :3, :3]
                Tmat_obj_time = rtks_objs_torch[obj_index][i : i+1, :3, 3]

            verts_obj_time = obj_to_cam(verts_obj_time, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
            mesh_obj_time = trimesh.Trimesh(vertices=np.asarray(verts_obj_time[0,:,:3].cpu()), faces=np.asarray(faces_obj_time[0].cpu()), vertex_colors=mesh_obj_time.visual.vertex_colors)
            meshr_obj_time = Mesh.from_trimesh(mesh_obj_time, smooth=True)
            meshr_obj_time._primitives[0].material.RoughnessFactor=1.

            if obj_index == len(opts_list) - 1 and opts.prealign:
                bkgd_meshes.append(meshr_obj_time)                        

            # 3. add the resulting mesh, where vertices are no defined in the desired camera frame to a Scene object
            scene.add_node(Node(mesh=meshr_obj_time))

        if opts.filter_3d or opts.render_cam:
            # 1. load foreground mesh
            meshdir_fg = opts_list[opts.asset_obj_index].rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
            meshdir_fg_time = glob.glob(meshdir_fg + "*-mesh-%05d.obj"%(frame_idx))
            assert(len(meshdir_fg_time) == 1)
            mesh_fg_time = trimesh.load(meshdir_fg_time[0], process=False)

            # transform vertices of 3d asset to fg frame using "asset2fg" transformation
            asset2fg = np.eye(4)

            if opts.filter_3d:
                # translation component = coordinate of the designated vertex in the fgframe
                asset2fg[:3, 3] = mesh_fg_time.vertices[opts.fg_normalbase_vertex_index, :].copy()

                # rotation component:
                # assume unicorn horn and camera both have OpenGL convention (where the normal / principal viewing direction points towards -ve z-axis)
                # extracting Z-axis direction of the 3d asset = vertex normal
                asset2fg[:3, 2] = -mesh_fg_time.vertex_normals[opts.fg_normalbase_vertex_index, :].copy()           # unit length normal vector: in OpenGL, z-axis is in opposite direction to OpenCV format

                # extracting X-axis = (downward vector in YZ plane = "y_prime") x Z-axis basis vector
                y_prime = mesh_fg_time.vertices[opts.fg_downdir_vertex_index, :] - mesh_fg_time.vertices[opts.fg_normalbase_vertex_index, :]
                yprime_cross_z = np.cross(y_prime, asset2fg[:3, 2])
                yprime_cross_z = yprime_cross_z / np.linalg.norm(yprime_cross_z)
                asset2fg[:3, 0] = yprime_cross_z

                # extracting Y-axis
                z_cross_x = np.cross(asset2fg[:3, 2], asset2fg[:3, 0])
                z_cross_x = z_cross_x / np.linalg.norm(z_cross_x)
                asset2fg[:3, 1] = -z_cross_x            # in OpenGL, y-axis is in opposite direction to OpenCV format
                asset2fg[:3, 3] = asset2fg[:3, 3] - opts.asset_offset_z * opts_list[-1].dep_scale * asset2fg[:3, 2]

                '''
                # rotation component:
                # extracting the Y-axis (the Y-axis denotes the principal direction of the 3d asset)
                # 2nd column: normal = fg_mesh.vertex_normal[chosen_vertex, :]
                asset2fg[:3, 1] = mesh_fg_time.vertex_normals[opts.fg_normalbase_vertex_index, :].copy()           # unit length normal vector

                # 1st column: [-R_23, 0, R_12].T where 2nd column = [R_12, R_22, R_23].T
                # since we just need the 1st column to perpendicular to the 2nd column
                asset2fg[0, 0] = -asset2fg[2, 1]
                asset2fg[1, 0] = 0.
                asset2fg[2, 0] = asset2fg[0, 1]

                # 3rd column
                cross_prod = np.cross(asset2fg[:3, 0], asset2fg[:3, 1])
                cross_prod = cross_prod / np.linalg.norm(cross_prod)
                asset2fg[:3, 2] = cross_prod
                '''
            if opts.render_cam:
                # translation component = coordinate of the camera frame origin in the fgframe (novelcam2fg)                
                fg2novelcam_time = rtks_objs[opts.asset_obj_index][i, ...].copy()
                fg2novelcam_time[:, 3] = fg2novelcam_time[:, 3]
                fg2novelcam_time[3, :] = np.array([0, 0, 0, 1]) 
                novelcam2fg_time = np.linalg.inv(fg2novelcam_time)

                asset2fg[:3, 3] = novelcam2fg_time[:3, 3]     # rtks_obj represents obj2novelcam transformation

                # rotation component:
                # assume unicorn horn and camera both have OpenGL convention (where the normal / principal viewing direction points towards -ve z-axis)
                # extracting Z-axis direction of the 3d asset = vertex normal
                asset2fg[:3, 2] = -novelcam2fg_time[:3, 2]       # in OpenGL (which is the convention the camera asset is defined in ), y-axis is in opposite direction to OpenCV format (which is the convention rtks_obj is defined in)

                # extracting X-axis
                asset2fg[:3, 0] = novelcam2fg_time[:3, 0]

                # extracting Y-axis
                asset2fg[:3, 1] = -novelcam2fg_time[:3, 1]       # in OpenGL, y-axis is in opposite direction to OpenCV format
                asset2fg[:3, 3] = asset2fg[:3, 3] - opts.asset_offset_z * opts_list[-1].dep_scale * asset2fg[:3, 2]

            asset_vertices_assetframe_homo = np.concatenate([mesh_asset.vertices, np.ones_like(mesh_asset.vertices[:, 0:1])], axis = -1)        # (N, 3) + (N, 1) = (N, 4)
            asset_vertices_fgframe_homo = np.matmul(asset2fg, asset_vertices_assetframe_homo.T).T

            # 4. change asset_mesh and fg_mesh from fg frame to camera frame
            #    if opts.render_camera, then we change to the camera frame specified by rtks_objs_rendercam_torch
            faces_asset_time = torch.Tensor(mesh_asset.faces[None]).cuda()
            verts_asset_time = torch.Tensor(asset_vertices_fgframe_homo[:, :3][None]).cuda()
            
            if opts.render_cam:
                Rmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, :3]
                Tmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, 3]
                #Rmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, :3]
                #Tmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, 3]
            else:
                Rmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, :3]
                Tmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, 3]

            verts_asset_time = obj_to_cam(verts_asset_time, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
            mesh_asset_time = trimesh.Trimesh(vertices=np.asarray(verts_asset_time[0,:,:3].cpu()), faces=np.asarray(faces_asset_time[0].cpu()), vertex_colors=mesh_asset.visual.vertex_colors.vertex_colors)     #
            
            #print("mesh_asset.visual.vertex_colors: {}".format(mesh_asset_time.visual.vertex_colors))

            meshr_asset_time = Mesh.from_trimesh(mesh_asset_time, smooth=True)
            #meshr_asset_time._primitives[0].material.RoughnessFactor=1.

            # 5. add the resulting mesh, where vertices are now defined in the desired camera frame to a Scene object
            scene.add_node(Node(mesh=meshr_asset_time))

        # 4. add a camera node with an I_4x4 transformation (adjusted for sign conventions)
        if opts.render_cam:
            focal_time = rtks_objs_rendercam_torch[-1][i, 3, :2]
            ppoint_time = rtks_objs_rendercam_torch[-1][i, 3, 2:]
            #focal_time = rtks_objs_torch[-1][i, 3, :2]
            #ppoint_time = rtks_objs_torch[-1][i, 3, 2:]
        else: 
            focal_time = rtks_objs_torch[-1][i, 3, :2]
            ppoint_time = rtks_objs_torch[-1][i, 3, 2:]
        cam_time = IntrinsicsCamera(
            focal_time[0],
            focal_time[1],
            ppoint_time[0],
            ppoint_time[1],
            znear=1e-3,zfar=1000)
        cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
        cam_node = scene.add(cam_time, pose=cam_pose)

        # 5. add a light source to the scene
        birdeyelight2fg = np.eye(4)
        birdeyelight2fg[:3, :3] = cv2.Rodrigues(np.asarray([-np.pi/2., 0., 0.]))[0]
        
        birdeyelight2fg[0, 3] = opts.topdowncam_offset_x                                                # offset along the x-direction of the fgroot frame for topdown view
        birdeyelight2fg[1, 3] = opts.topdowncam_offset_y                                                # offset along the y-direction of the fgroot frame for topdown view
        birdeyelight2fg[2, 3] = opts.topdowncam_offset_z
        
        if opts.fix_frame < 0:
            i_fixframe = opts.fix_frame
        else:
            i_fixframe = np.argmax(sample_idx == opts.fix_frame)
        if opts.render_cam:
            bkgd2cam_fixframe = np.concatenate([rtks_objs_rendercam[-1][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
            fg2cam_fixframe = np.concatenate([rtks_objs_rendercam[opts.fg_obj_index][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
            bkgd2cam_time = np.concatenate([rtks_objs_rendercam[-1][i, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
        else:
            bkgd2cam_fixframe = np.concatenate([rtks_objs[-1][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
            fg2cam_fixframe = np.concatenate([rtks_objs[opts.fg_obj_index][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
            bkgd2cam_time = np.concatenate([rtks_objs[-1][i, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)

        #print("opts.fix_frame: {}".format(opts.fix_frame))      # fix_frame = -1
        #print("frame_idx: {}".format(frame_idx))
        #print("i: {}".format(i))

        fgfixframe2bkgd = np.matmul(np.linalg.inv(bkgd2cam_fixframe), fg2cam_fixframe)
        birdeyelight2bkgd = np.matmul(fgfixframe2bkgd, birdeyelight2fg)
        birdeyelight2cam = np.matmul(bkgd2cam_time, birdeyelight2bkgd)


        # represents the birdeyelight pose in camera space
        #print("birdeyelight2fg: {}".format(birdeyelight2fg))
        #print("")

        #birdeyelight2cam = np.matmul(np.matmul(np.concatenate([rtks_objs[-1][frame_idx, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0), fgfixframe2bkgd), birdeyelight2fg)
        
        #theta = 9*np.pi/9
        #init_light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
        #light_pose = init_light_pose
        direc_l_node = scene.add(direc_l, pose=birdeyelight2cam)

        # 6. render Scene
        mesh_rnd_color_time, mesh_rnd_depth_time = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        #rndsil = (mesh_rnd_depth_time[:960,:720]>0).astype(int)*100
        mesh_rnd_color_time = mesh_rnd_color_time[:height, :width, :3]
        mesh_rnd_depth_time = mesh_rnd_depth_time[:height, :width]

        near = np.min(mesh_rnd_depth_time[mesh_rnd_depth_time > 0])
        far = np.max(mesh_rnd_depth_time[mesh_rnd_depth_time > 0])
        near_far = np.stack([near, far])
        print("near - far (m): {}, {}".format(near / opts_list[-1].dep_scale, far / opts_list[-1].dep_scale))
        near_far_values.append(near_far)

        cv2.imwrite('%s-mesh_%05d.png'%(opts.nvs_outpath,frame_idx), mesh_rnd_color_time[...,::-1])
        mesh_rnd_colors.append(mesh_rnd_color_time)

    save_vid('%s-mesh'%(opts.nvs_outpath), mesh_rnd_colors, suffix='.mp4', upsample_frame=-1, fps=10)
    # scale back to metric space (i.e. in meters)
    near_far_values = np.stack(near_far_values, axis = 0) / opts_list[-1].dep_scale                       # (size, 2)
    near_far_overallframes = [np.min(near_far_values[:, 0]), np.max(near_far_values[:, 1])]
    near_far_overallframes = np.stack(near_far_overallframes)
    np.save('%s-nf_perframe.npy'%(opts.nvs_outpath), near_far_values)
    np.save('%s-nf_overallframes.npy'%(opts.nvs_outpath), near_far_overallframes)
    print("near_far_overallframes: {}".format(near_far_overallframes))

    if opts.filter_3d:
        print("RENDERING 3D ASSET")
        # load asset mesh
        resolver = trimesh.resolvers.FilePathResolver('mesh_material/UnicornHorn_OBJ/')
        mesh_asset = trimesh.load(file_obj='mesh_material/UnicornHorn_OBJ/UnicornHorn.obj', resolver = resolver, process=False)
        color = mesh_asset.visual.to_color()
        mesh_asset.visual.vertex_colors = color
        mesh_asset.visual.vertex_colors.vertex_colors = np.tile(np.array([[0.439216, 0.9137255, 0.99607843, 1.]]), (mesh_asset.visual.vertex_colors.vertex_colors.shape[0], 1))

        # scale vertices of 3d asset
        mesh_asset.vertices = mesh_asset.vertices * opts.asset_scale

        meshasset_rnd_colors = []
        meshasset_rnd_depths = []
        for i, frame_idx in enumerate(sample_idx):    
            print("index {} / {}".format(i, len(sample_idx)))
            
            r = OffscreenRenderer(img_size, img_size)
            scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))

            # 1. load foreground mesh
            meshdir_fg = opts_list[opts.asset_obj_index].rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
            meshdir_fg_time = glob.glob(meshdir_fg + "*-mesh-%05d.obj"%(frame_idx))
            assert(len(meshdir_fg_time) == 1)
            mesh_fg_time = trimesh.load(meshdir_fg_time[0], process=False)

            # transform vertices of 3d asset to fg frame using "asset2fg" transformation
            asset2fg = np.eye(4)

            # translation component = coordinate of the designated vertex in the fgframe
            asset2fg[:3, 3] = mesh_fg_time.vertices[opts.fg_normalbase_vertex_index, :].copy()
            # 2nd column: normal = fg_mesh.vertex_normal[chosen_vertex, :]
            asset2fg[:3, 1] = mesh_fg_time.vertex_normals[opts.fg_normalbase_vertex_index, :].copy()           # unit length normal vector

            # 1st column: [-R_32, 0, R_12].T where 2nd column = [R_12, R_22, R_23].T
            asset2fg[0, 0] = -asset2fg[2, 1]
            asset2fg[1, 0] = 0.
            asset2fg[2, 0] = asset2fg[0, 1]

            # 3rd column
            cross_prod = np.cross(asset2fg[:3, 0], asset2fg[:3, 1])
            cross_prod = cross_prod / np.linalg.norm(cross_prod)
            asset2fg[:3, 2] = cross_prod

            asset_vertices_assetframe_homo = np.concatenate([mesh_asset.vertices, np.ones_like(mesh_asset.vertices[:, 0:1])], axis = -1)        # (N, 3) + (N, 1) = (N, 4)
            asset_vertices_fgframe_homo = np.matmul(asset2fg, asset_vertices_assetframe_homo.T).T

            # 4. change asset_mesh and fg_mesh from fg frame to camera frame
            faces_asset_time = torch.Tensor(mesh_asset.faces[None]).cuda()
            verts_asset_time = torch.Tensor(asset_vertices_fgframe_homo[:, :3][None]).cuda()
            Rmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, :3]
            Tmat_obj_time = rtks_objs_torch[opts.asset_obj_index][i : i+1, :3, 3]

            verts_asset_time = obj_to_cam(verts_asset_time, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
            mesh_asset_time = trimesh.Trimesh(vertices=np.asarray(verts_asset_time[0,:,:3].cpu()), faces=np.asarray(faces_asset_time[0].cpu()), vertex_colors=mesh_asset.visual.vertex_colors.vertex_colors)     #
            
            #print("mesh_asset.visual.vertex_colors: {}".format(mesh_asset_time.visual.vertex_colors))

            meshr_asset_time = Mesh.from_trimesh(mesh_asset_time, smooth=True)
            #meshr_asset_time._primitives[0].material.RoughnessFactor=1.

            # 5. add the resulting mesh, where vertices are now defined in the desired camera frame to a Scene object
            scene.add_node(Node(mesh=meshr_asset_time))

            # 6. add a camera node with an I_4x4 transformation (adjusted for sign conventions)
            focal_time = rtks_objs_torch[-1][i, 3, :2]
            ppoint_time = rtks_objs_torch[-1][i, 3, 2:]
            cam_time = IntrinsicsCamera(
                focal_time[0],
                focal_time[1],
                ppoint_time[0],
                ppoint_time[1],
                znear=1e-3,zfar=1000)
            cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
            cam_node = scene.add(cam_time, pose=cam_pose)

            # 5. add a light source to the scene
            theta = 9*np.pi/9
            init_light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
            light_pose = init_light_pose
            direc_l_node = scene.add(direc_l, pose=light_pose)

            # 6. render Scene
            meshasset_rnd_color_time, meshasset_rnd_depth_time = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
            r.delete()
            meshasset_rnd_color_time = meshasset_rnd_color_time[:height, :width, :3]
            meshasset_rnd_depth_time = meshasset_rnd_depth_time[:height, :width]
            cv2.imwrite('%s-meshasset_%05d.png'%(opts.nvs_outpath,frame_idx), meshasset_rnd_color_time[...,::-1])
            #cv2.imwrite('%s-mesh_%05d.png'%(opts.nvs_outpath,i), mesh_rnd_color_time[...,::-1])
            meshasset_rnd_colors.append(meshasset_rnd_color_time)
            meshasset_rnd_depths.append(meshasset_rnd_depth_time)

    if opts.prealign:
        print("MESH RENDERING BKGD FOR PREALIGNING")
        for i, frame_idx in enumerate(sample_idx):    
            print("index {} / {}".format(i, len(sample_idx)))
            r = OffscreenRenderer(img_size, img_size)
            scene = Scene(ambient_light=0.6*np.asarray([1.,1.,1.,1.]))
            
            meshr_bkgd_time = bkgd_meshes[i]
            
            # add the bkgd mesh, where vertices are no defined in the desired camera frame to a Scene object
            scene.add_node(Node(mesh=meshr_bkgd_time))

            # add camera to scene
            if opts.render_cam:
                focal_time = rtks_objs_rendercam_torch[-1][i, 3, :2]
                ppoint_time = rtks_objs_rendercam_torch[-1][i, 3, 2:]
                #focal_time = rtks_objs_torch[-1][i, 3, :2]
                #ppoint_time = rtks_objs_torch[-1][i, 3, 2:]
            else: 
                focal_time = rtks_objs_torch[-1][i, 3, :2]
                ppoint_time = rtks_objs_torch[-1][i, 3, 2:]
            cam_time = IntrinsicsCamera(
                focal_time[0],
                focal_time[1],
                ppoint_time[0],
                ppoint_time[1],
                znear=1e-3,zfar=1000)
            cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
            cam_node = scene.add(cam_time, pose=cam_pose)

            # 
            _, bkgd_mesh_rnd_depth_time = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
            r.delete()
            #rndsil = (mesh_rnd_depth_time[:960,:720]>0).astype(int)*100
            bkgd_mesh_rnd_depth_time = bkgd_mesh_rnd_depth_time[:height, :width]
            bkgd_mesh_rnd_depths.append(bkgd_mesh_rnd_depth_time)

    near_far_objs = []
    for obj_index, obj in enumerate(model.objs):
        vars_np = {}
        vars_np['rtk'] = rtks_objs[obj_index]
        vars_np['idk'] = np.ones(bs)
        near_far = torch.zeros(bs,2).to(model.device)
        near_far = get_near_far(near_far,
                                vars_np,
                                pts=obj.latest_vars['mesh_rest'].vertices)
        near_far_objs.append(near_far)

    vidid = torch.Tensor([opts.vidid]).to(model.device).long()
    #source_l = model.data_offset[opts.vidid+1] - model.data_offset[opts.vidid] -1
    #embedid = torch.Tensor(sample_idx).to(model.device).long() + \
    #          model.data_offset[opts.vidid]
    #if opts.bullet_time>-1: embedid[:] = opts.bullet_time+model.data_offset[opts.vidid]
    source_l = model.objs[-1].data_offset[opts.vidid+1] - model.objs[-1].data_offset[opts.vidid] -1
    embedid = torch.Tensor(sample_idx).to(model.device).long() + \
              model.objs[-1].data_offset[opts.vidid]
    if opts.bullet_time>-1: embedid[:] = opts.bullet_time+model.objs[-1].data_offset[opts.vidid]

    rgbs_gt_refcam = []
    dphs_gt_refcam = []
    confs_gt_refcam = []
    # multi-obj case
    #sils_gt_refcam = []
    sils_gt_refcam_objs = {obj_index: [] for obj_index in range(len(opts_list))}

    rgbs_gt_secondcam = []
    dphs_gt_secondcam = []
    confs_gt_secondcam = []
    # multi-obj case
    #sils_gt_secondcam = []
    sils_gt_secondcam_objs = {obj_index: [] for obj_index in range(len(opts_list))}

    rgbs = []
    sils = []
    sils_objs = {obj_index: [] for obj_index in range(len(opts_list))}  
    viss = []
    dphs = []

    if opts.filter_3d:
        rgbs_with_asset = []

    if opts.evaluate:
        lpips_model = lpips_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True,version=0.1)
        chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

        x_coord, y_coord = np.meshgrid(np.arange(width), np.arange(height))                             # (H, W)
        p_homogen = np.stack([x_coord, y_coord, np.ones_like(y_coord)], axis = -1)                      # (H, W, 3)

        rgb_error_maps = []
        rgb_errors = []                                                                                 # error averaged over the entire image
        rgb_errors_minus_holes = []
        psnrs = []
        ssims = []
        lpipss = []
        cds = []
        cds_minus_holes = []
        f_at_5cms = []
        f_at_5cms_minus_holes = []
        f_at_10cms = []
        f_at_10cms_minus_holes = []
        depth_error_maps = []
        depth_errors = []                                                                               # error averaged over the entire image
        depth_errors_minus_holes = []
        
        rgb_errors_objs = {obj_index: [] for obj_index in range(len(opts_list))}                        # error averaged over the rendered silhouette of each object
        psnrs_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        ssims_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        lpipss_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        cds_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        f_at_5cms_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        f_at_10cms_objs = {obj_index: [] for obj_index in range(len(opts_list))}
        depth_errors_objs = {obj_index: [] for obj_index in range(len(opts_list))}                      # error averaged over the rendered silhouette of each object

    # for i in range(bs):
    for i, frame_index in enumerate(sample_idx):
        print("synthesizing frame {}".format(i))

        rndsil = rndsils[i]
        rndmask = np.zeros((img_size, img_size))

        rgb_gt_refcam = cv2.imread(gt_imglist_refcam[frame_index])[:,:,::-1] / 255.0                # BGR -> RGB
        dph_gt_refcam = read_depth(gt_deplist_refcam[frame_index])
        conf_gt_refcam = read_conf(gt_conflist_refcam[frame_index])
        
        #multi-obj case
        #sil_gt_refcam = cv2.imread(gt_masklist_refcam[frame_index], 0)                             # shape = (H, W)
        sil_gt_refcam_objs = [cv2.imread(gt_masklist_refcam[frame_index], 0) for gt_masklist_refcam in gt_masklist_refcam_objs]                 # list of arrays of shape = (H, W)
        
        # filter the silhouettes
        for obj_index, sil_gt_refcam_obj in enumerate(sil_gt_refcam_objs):
            #print("obj {}: unique mask elements before filtering {}".format(obj_index, np.unique(sil_gt_refcam_obj)))
            
            # if the mask represents a frame with an invalid mask, it will have mask.nelements - 1 entries of 255
            if sil_gt_refcam_obj[sil_gt_refcam_obj == 255].size == sil_gt_refcam_obj.size - 1:
                sil_gt_refcam_obj[sil_gt_refcam_obj != 255] = 255             # we set the dummy pixel whose value was set to 1 so as not to induce an error in autogen.py and vidbase.py   
            
            # either a pixel can be 
            # 0:    bkgd pixel
            # 1:    fg pixel
            # 254:  belonging to another fg
            # 255:  pixel belonging to frame without a valid mask
            sil_gt_refcam_obj = np.where(sil_gt_refcam_obj < 200, (sil_gt_refcam_obj>0).astype(float), sil_gt_refcam_obj)

            # for bkgd, invert the silhouette
            if opts_list[obj_index].recon_bkgd:
                sil_gt_refcam_obj = np.where(sil_gt_refcam_obj < 200, 1 - sil_gt_refcam_obj, sil_gt_refcam_obj)

            #print("obj {}: unique mask elements after filtering {}".format(obj_index, np.unique(sil_gt_refcam_obj)))

            sil_gt_refcam_objs[obj_index] = sil_gt_refcam_obj

        conf_gt_refcam[np.isnan(dph_gt_refcam)] = 0.
        dph_gt_refcam[np.isnan(dph_gt_refcam)] = 4.
        dph_gt_refcam = dph_gt_refcam * opts_list[-1].dep_scale                                     # max depth recorded by the LiDAR is 4m
        assert (not np.any(np.isnan(dph_gt_refcam)))

        #TODO: upsample depth map "dep" to imgsize
        #if dph_gt.shape[0] != rgb_gt.shape[0] or dph_gt.shape[1]!= rgb_gt.shape[1]:
        #    dph_gt = cv2.resize(dph_gt, rgb_gt.shape[:2][::-1],interpolation=cv2.INTER_LINEAR)

        if opts.stereo_view:
            rgb_gt_secondcam = cv2.imread(gt_imglist_secondcam[frame_index])[:,:,::-1] / 255.0      # BGR -> RGB
            dph_gt_secondcam = read_depth(gt_deplist_secondcam[frame_index])
            conf_gt_secondcam = read_conf(gt_conflist_secondcam[frame_index])
            
            #multi-obj case
            #sil_gt_secondcam = cv2.imread(gt_masklist_secondcam[frame_index], 0)                   # shape = (H, W)
            sil_gt_secondcam_objs = [cv2.imread(gt_masklist_secondcam[frame_index], 0) for gt_masklist_secondcam in gt_masklist_secondcam_objs]      # list of arrays of shape = (H, W)
            
            # filter the silhouettes
            for obj_index, sil_gt_secondcam_obj in enumerate(sil_gt_secondcam_objs):
                # if the mask represents a frame with an invalid mask, it will have mask.nelements - 1 entries of 255
                if sil_gt_secondcam_obj[sil_gt_secondcam_obj == 255].size == sil_gt_secondcam_obj.size - 1:
                    sil_gt_secondcam_obj[sil_gt_secondcam_obj != 255] = 255             # we set the dummy pixel whose value was set to 1 so as not to induce an error in autogen.py and vidbase.py   
                
                # either a pixel can be 
                # 0:    bkgd pixel
                # 1:    fg pixel
                # 254:  belonging to another fg
                # 255:  pixel belonging to frame without a valid mask
                sil_gt_secondcam_obj = np.where(sil_gt_secondcam_obj < 200, (sil_gt_secondcam_obj>0).astype(float), sil_gt_secondcam_obj)

                # for bkgd, invert the silhouette
                if opts_list[obj_index].recon_bkgd:
                    sil_gt_secondcam_obj = np.where(sil_gt_secondcam_obj < 200, 1 - sil_gt_secondcam_obj, sil_gt_secondcam_obj)

                sil_gt_secondcam_objs[obj_index] = sil_gt_secondcam_obj

            conf_gt_secondcam[np.isnan(dph_gt_secondcam)] = 0.
            dph_gt_secondcam[np.isnan(dph_gt_secondcam)] = 4.
            dph_gt_secondcam = dph_gt_secondcam * opts_list[-1].dep_scale                 # max depth recorded by the LiDAR is 4m 
            
            assert (not np.any(np.isnan(dph_gt_secondcam)))

            #TODO: upsample depth map "dep" to imgsize
            #if dph_gt_secondcam.shape[0] != rgb_gt_secondcam.shape[0] or dph_gt_secondcam.shape[1]!= rgb_gt_secondcam.shape[1]:
            #    dph_gt_secondcam = cv2.resize(dph_gt_secondcam, rgb_gt_secondcam.shape[:2][::-1],interpolation=cv2.INTER_LINEAR)
            #    conf_gt_secondcam = cv2.resize(conf_gt_secondcam, rgb_gt_secondcam.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)

        if img_type=='vert':
            size_short_edge = int(rndsil.shape[1] * img_size/rndsil.shape[0])
            rndsil = cv2.resize(rndsil, (size_short_edge, img_size))
            rndmask[:,:size_short_edge] = rndsil

            rgb_gt_refcam = cv2.resize(rgb_gt_refcam, (size_short_edge, img_size))
            dph_gt_refcam = cv2.resize(dph_gt_refcam, (size_short_edge, img_size), interpolation=cv2.INTER_LINEAR)
            conf_gt_refcam = cv2.resize(conf_gt_refcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST)
            
            # multi-obj case
            #sil_gt_refcam = cv2.resize(sil_gt_refcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST)
            sil_gt_refcam_objs = [cv2.resize(sil_gt_refcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST) for sil_gt_refcam in sil_gt_refcam_objs]

            if opts.stereo_view:
                rgb_gt_secondcam = cv2.resize(rgb_gt_secondcam, (size_short_edge, img_size))
                dph_gt_secondcam = cv2.resize(dph_gt_secondcam, (size_short_edge, img_size), interpolation=cv2.INTER_LINEAR)
                conf_gt_secondcam = cv2.resize(conf_gt_secondcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST)
                
                # multi-obj case
                #sil_gt_secondcam = cv2.resize(sil_gt_secondcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST)
                sil_gt_secondcam_objs = [cv2.resize(sil_gt_secondcam, (size_short_edge, img_size), interpolation=cv2.INTER_NEAREST) for sil_gt_secondcam in sil_gt_secondcam_objs]

        else:
            size_short_edge = int(rndsil.shape[0] * img_size/rndsil.shape[1])
            rndsil = cv2.resize(rndsil, (img_size, size_short_edge))
            rndmask[:size_short_edge] = rndsil

            rgb_gt_refcam = cv2.resize(rgb_gt_refcam, (img_size, size_short_edge))
            dph_gt_refcam = cv2.resize(dph_gt_refcam, (img_size, size_short_edge), interpolation=cv2.INTER_LINEAR)
            conf_gt_refcam = cv2.resize(conf_gt_refcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST)
            
            # multi-obj case
            #sil_gt_refcam = cv2.resize(sil_gt_refcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST)
            sil_gt_refcam_objs = [cv2.resize(sil_gt_refcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST) for sil_gt_refcam in sil_gt_refcam_objs]

            if opts.stereo_view:
                rgb_gt_secondcam = cv2.resize(rgb_gt_secondcam, (img_size, size_short_edge))
                dph_gt_secondcam = cv2.resize(dph_gt_secondcam, (img_size, size_short_edge), interpolation=cv2.INTER_LINEAR)
                conf_gt_secondcam = cv2.resize(conf_gt_secondcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST)
                
                # multi-obj case
                #sil_gt_secondcam = cv2.resize(sil_gt_secondcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST)
                sil_gt_secondcam_objs = [cv2.resize(sil_gt_secondcam, (img_size, size_short_edge), interpolation=cv2.INTER_NEAREST) for sil_gt_secondcam in sil_gt_secondcam_objs]

        # apply pre-alignment between the estimated mesh and the gt depth for the stereo-view to properly evaluate non-depth supervised results
        if opts.prealign:
            # a) compute pointcloud for estimated mesh:
            #    backproject the mesh-rendered depth into the "estimated" stereo-view camera space
            #
            
            bkgd_mesh_rnd_depth_time = bkgd_mesh_rnd_depths[i]        # shape = (height, width); current frame index is given by "i"
            focal_refcam = rtks_objs[-1][i, 3, :2]
            ppoint_refcam = rtks_objs[-1][i, 3, 2:]
            K_bkgd_mesh   = np.array([[focal_refcam[0], 0., ppoint_refcam[0]],                                                                            
                                    [0., focal_refcam[1], ppoint_refcam[1]],
                                    [0., 0., 1.]])
            P = np.repeat(bkgd_mesh_rnd_depth_time[..., np.newaxis], 3, axis=-1) * np.matmul(p_homogen, np.repeat(np.linalg.inv(K_bkgd_mesh.T)[np.newaxis, ...], height, axis = 0))       # (H, W, 3); for np.matmul, if either argument is N-D, it's treated as a stack of matrices residing in the last two indexes
            P = torch.from_numpy(P.astype(np.float32)).cuda()

            # filter out points with zero mesh-rendered depth
            x_coord_valid = x_coord[bkgd_mesh_rnd_depth_time > 0]                                          # (H, W)[(H, W)]                                                             
            y_coord_valid = y_coord[bkgd_mesh_rnd_depth_time > 0]                                          # (H, W)[(H, W)]
            P_valid = P[y_coord_valid, x_coord_valid, :]

            # b) compute pointcloud based on gt depth:
            #    backproject the gt, stereo-view depth to the stereo view camera space
            P_gt = np.repeat(dph_gt_secondcam[..., np.newaxis], 3, axis=-1) * np.matmul(p_homogen, np.repeat(np.linalg.inv(K_secondcam.T)[np.newaxis, ...], height, axis = 0))       # (H, W, 3); for np.matmul, if either argument is N-D, it's treated as a stack of matrices residing in the last two indexes
            P_gt = torch.from_numpy(P_gt.astype(np.float32)).cuda()
            
            # filter out points with 1) low confidence and 2) that don't belong to the bkgd
            x_coord_valid_gt = x_coord[(conf_gt_secondcam > 1.5) & (sil_gt_secondcam_objs[-1] == 1)]                                          # (H, W)[(H, W)]                                                             
            y_coord_valid_gt = y_coord[(conf_gt_secondcam > 1.5) & (sil_gt_secondcam_objs[-1] == 1)]                                          # (H, W)[(H, W)]
            
            # if num. of remaining points is non-zero 
            if not np.any(np.isnan(x_coord_valid_gt)):
                P_gt_valid = P_gt[y_coord_valid_gt, x_coord_valid_gt, :]                            # (N_valid, 3) 

                # compute alignment between estimated stereo-view and target stereo-view
                fitted_scale = P_gt_valid[...,-1].median() / P_valid[...,-1].median()
                P_valid = P_valid*fitted_scale

                frts = pytorch3d.ops.iterative_closest_point(P_valid[None, ...], P_gt_valid[None, ...], \
                        estimate_scale=False,max_iterations=100)

                prealign_srt = np.eye(4)
                prealign_srt[:3, :3] = (fitted_scale * frts.RTs.s[0] * frts.RTs.R[0, ...]).cpu().numpy()
                prealign_srt[:3, 3] = (fitted_scale * frts.RTs.T[0, ...]).cpu().numpy()

                print("fitted_scale: {}".format(fitted_scale))
                print("prealign_srt: {}".format(prealign_srt))

                # offset camera poses for all objects by prealign_srt
                for obj_index in range(len(rtks_objs)):
                    print("[BEFORE]: {}".format(rtks_objs[obj_index][i, ...]))
                    rts_bkgd_i = np.eye(4)
                    rts_bkgd_i[:3, :] = rtks_objs[obj_index][i, :3, :].copy()
                    rts_bkgd_i = np.matmul(prealign_srt, rts_bkgd_i)            # (4, 4) = (4, 4) * (4, 4)
                    rtks_objs[obj_index][i, :3, :] = rts_bkgd_i[:3, :]
                    print("[AFTER]: {}".format(rtks_objs[obj_index][i, ...]))

        rays_objs = []
        for obj_index, obj in enumerate(model.objs):
            rays = construct_rays_nvs(model.img_size, rtks_objs[obj_index][i:i+1], 
                                       near_far_objs[obj_index][i:i+1], rndmask, model.device)
            
            # add env code
            rays['env_code'] = obj.env_code(embedid[i:i+1])[:,None]
            rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)

            # add bones
            time_embedded = obj.pose_code(embedid[i:i+1])[:,None]
            rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
            if opts_list[obj_index].lbs and obj.num_bone_used>0:
                bone_rts = obj.nerf_body_rts(embedid[i:i+1])
                rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
                obj.update_delta_rts(rays)
            
            rays_objs.append(rays)

        with torch.no_grad():
            # render images only
            results=defaultdict(list)
            bs_rays = rays_objs[-1]['bs'] * rays_objs[-1]['nsample'] # over pixels

            for j in range(0, bs_rays, opts.chunk):
                rays_chunk_objs = []
            
                for rays in rays_objs:
                    rays_chunk = chunk_rays(rays,j,opts.chunk)
                    rays_chunk_objs.append(rays_chunk)

                rendered_chunks = render_rays_objs([obj.nerf_models for obj in model.objs],
                        [obj.embeddings for obj in model.objs],
                        rays_chunk_objs,
                        N_samples = opts.ndepth,
                        perturb=0,
                        noise_std=0,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        obj_bounds=[obj.latest_vars['obj_bound'] for obj in model.objs],
                        use_fine=True,
                        img_size=model.img_size,
                        render_vis=True,
                        opts=opts,
                        opts_objs=opts_list
                        )

                for k, v in rendered_chunks.items():
                    results[k] += [v]
           
        for k, v in results.items():
            v = torch.cat(v, 0)
            v = v.view(rays['nsample'], -1)
            results[k] = v

        rgb = results['img_coarse'].cpu().numpy()
        dph = results['depth_rnd'] [...,0].cpu().numpy()
        sil = results['sil_coarse'][...,0].cpu().numpy()
        vis = results['vis_pred']  [...,0].cpu().numpy()

        sil_objs = []
        for obj_index in range(len(opts_list)):
            sil_obj = results['sil_coarse_obj{}'.format(obj_index)][...,0].cpu().numpy()
            #sil_obj[sil_obj < 0.5] = 0
            sil_objs.append(sil_obj)

        #sil[sil<0.5] = 0
        dph[sil<0.5] = 0
        rgb[sil<0.5] = 1

        rgbtmp = np.ones((img_size, img_size, 3))
        dphtmp = np.ones((img_size, img_size))
        siltmp = np.ones((img_size, img_size))
        vistmp = np.ones((img_size, img_size))
        rgbtmp[rndmask>0] = rgb
        dphtmp[rndmask>0] = dph
        siltmp[rndmask>0] = sil
        vistmp[rndmask>0] = vis

        if img_type=='vert':
            rgb = rgbtmp[:,:size_short_edge]
            sil = siltmp[:,:size_short_edge]
            vis = vistmp[:,:size_short_edge]
            dph = dphtmp[:,:size_short_edge]
        else:
            rgb = rgbtmp[:size_short_edge]
            sil = siltmp[:size_short_edge]
            vis = vistmp[:size_short_edge]
            dph = dphtmp[:size_short_edge]
    
        for obj_index, sil_obj in enumerate(sil_objs):
            siltmp_obj = np.ones((img_size, img_size))
            siltmp_obj[rndmask>0] = sil_obj
            if img_type=='vert':
                sil_objs[obj_index] = siltmp_obj[:,:size_short_edge]
            else:
                sil_objs[obj_index] = siltmp_obj[:size_short_edge]

        # 3D FILTER
        if opts.filter_3d:
            rgb_asset = meshasset_rnd_colors[i] / 255.         # \in [0, 255]
            dph_asset = meshasset_rnd_depths[i]

            rgb_asset_mask = (dph_asset > 0) & (dph_asset < dph)
            rgb_with_asset = rgb_asset_mask[..., None] * rgb_asset + (1. - rgb_asset_mask[..., None]) * rgb_gt_refcam       # overlay onto raw RGB
            rgbs_with_asset.append(rgb_with_asset)

        # COMPUTING EVALUATION METRICS
        if opts.evaluate:
            if opts.stereo_view:
                #rgb_diff = (rgb - rgb_gt_secondcam) * (vis[..., np.newaxis] > 0.5)
                #depth_diff = (dph - dph_gt_secondcam) / opts_list[-1].dep_scale * (vis > 0.5) * (conf_gt_secondcam > 1.5)
                rgb_gt = rgb_gt_secondcam
                dph_gt = dph_gt_secondcam
                conf_gt = conf_gt_secondcam 
                sil_gt_objs = sil_gt_secondcam_objs

                K = K_secondcam
                print("K_stereoview: {}".format(K))                           

            if opts.input_view:
                rgb_gt = rgb_gt_refcam
                dph_gt = dph_gt_refcam
                conf_gt = conf_gt_refcam
                sil_gt_objs = sil_gt_refcam_objs

                focal_refcam = rtks_objs[-1][i, 3, :2]
                ppoint_refcam = rtks_objs[-1][i, 3, 2:]
                K       = np.array([[focal_refcam[0], 0., ppoint_refcam[0]],                                                                            
                                    [0., focal_refcam[1], ppoint_refcam[1]],
                                    [0., 0., 1.]])

            if opts.stereo_view or opts.input_view:
                rgb_diff = (rgb - rgb_gt)                                                                   # rgb \in [0, 1]
                depth_diff = (dph - dph_gt) / opts_list[-1].dep_scale * (conf_gt > 1.5)                     # we only consider pixels whose confidence for gt depth is high    

                # MSE (units: m^2)
                rgb_error = np.mean(np.power(rgb_diff, 2))                                                  # shape = (H, W, C)
                #depth_error = np.mean(np.power(depth_diff, 2))                                              # shape = (H, W)
                depth_error = compute_depth_error(dph_gt, dph, conf_gt, mask=None, dep_scale=opts_list[-1].dep_scale)
                
                #rgb_error_minus_holes = np.mean(np.power(rgb_diff[sil > 0.5], 2))                           # shape = (H, W, C)
                #depth_error_minus_holes = np.mean(np.power(depth_diff[sil > 0.5], 2))                       # shape = (H, W)
                rgb_errors.append(rgb_error)
                depth_errors.append(depth_error)
                #rgb_errors_minus_holes.append(rgb_error_minus_holes)
                #depth_errors_minus_holes.append(depth_error_minus_holes)

                # Per-pixel, absolute error (units: m)
                rgb_error_maps.append(np.sqrt(np.mean(np.power(rgb_diff, 2), axis = -1)))
                depth_error_maps.append(np.abs(depth_diff))
                
                # PSNR
                #psnr = peak_signal_noise_ratio(rgb_gt, rgb)                                                             # both rgb_gt and rgb need to be in range [0, 1]
                psnr = compute_psnr(rgb_gt, rgb)
                psnrs.append(psnr)

                # SSIM
                #ssim = structural_similarity(rgb_gt, rgb, channel_axis=-1)
                ssim = compute_ssim(rgb_gt, rgb)
                ssims.append(ssim)

                # LPIPS
                '''
                rgb_gt_0 = im2tensor(rgb_gt).cuda()
                rgb_0 = im2tensor(rgb).cuda()
                lpips = lpips_model.forward(rgb_gt_0, rgb_0)
                lpips = lpips.item()
                '''
                lpips = compute_lpips(rgb_gt, rgb, lpips_model)
                lpipss.append(lpips)

                # chamfer dist and F-score
                # backproject gt depth and rendered depth into pointclouds in camera space (units: m)
                # ASSUME: intrinsics for the fg and bkgd are the same
                # TO-DO: unify the intrinsics for the fg and bkgd
                '''
                P_gt = np.repeat(dph_gt[..., np.newaxis], 3, axis=-1) * np.matmul(p_homogen, np.repeat(np.linalg.inv(K.T)[np.newaxis, ...], height, axis = 0)) / opts_list[-1].dep_scale       # (H, W, 3); for np.matmul, if either argument is N-D, it's treated as a stack of matrices residing in the last two indexes
                P_gt = torch.from_numpy(P_gt.astype(np.float32)).cuda()
                P = np.repeat(dph[..., np.newaxis], 3, axis=-1) * np.matmul(p_homogen, np.repeat(np.linalg.inv(K.T)[np.newaxis, ...], height, axis = 0)) / opts_list[-1].dep_scale             # (H, W, 3)
                P = torch.from_numpy(P.astype(np.float32)).cuda()

                # filter out for points with low confidence
                x_coord_valid = x_coord[conf_gt > 1.5]                                          # (H, W)[(H, W)]                                                             
                y_coord_valid = y_coord[conf_gt > 1.5]                                          # (H, W)[(H, W)]
                #x_coord_valid_minus_holes = x_coord[(conf_gt > 1.5) & (sil > 0.5)]              # (H, W)[(H, W)]                                                             
                #y_coord_valid_minus_holes = y_coord[(conf_gt > 1.5) & (sil > 0.5)]              # (H, W)[(H, W)]

                P_gt_valid = P_gt[y_coord_valid, x_coord_valid, :]                                                  # (N_valid, 3)
                P_valid = P[y_coord_valid, x_coord_valid, :]                                                        # (N_valid, 3)
                #P_gt_valid_minus_holes = P_gt[y_coord_valid_minus_holes, x_coord_valid_minus_holes, :]              # (N_valid, 3)
                #P_valid_minus_holes = P[y_coord_valid_minus_holes, x_coord_valid_minus_holes, :]                    # (N_valid, 3)

                # compute metrics
                raw_cd, raw_cd_back, _, _ = chamLoss(P_gt_valid[None, ...], P_valid[None, ...])
                f_at_5cm, _, _ = fscore.fscore(raw_cd, raw_cd_back, threshold = 0.05**2)
                f_at_10cm, _, _ = fscore.fscore(raw_cd, raw_cd_back, threshold = 0.10**2)
                
                raw_cd = np.sqrt(np.asarray(raw_cd.cpu()[0]))
                raw_cd_back = np.sqrt(np.asarray(raw_cd_back.cpu()[0]))
                cd = raw_cd.mean() + raw_cd_back.mean()
                '''

                cd, f_at_5cm, f_at_10cm = compute_chamfer_dist_fscore(dph_gt, dph, conf_gt, K, chamLoss, fscore.fscore, mask=None, dep_scale=opts_list[-1].dep_scale)

                #raw_cd_minus_holes, raw_cd_back_minus_holes, _, _ = chamLoss(P_gt_valid_minus_holes[None, ...], P_valid_minus_holes[None, ...])
                #f_at_5cm_minus_holes, _, _ = fscore.fscore(raw_cd_minus_holes, raw_cd_back_minus_holes, threshold = 0.05**2)
                #f_at_10cm_minus_holes, _, _ = fscore.fscore(raw_cd_minus_holes, raw_cd_back_minus_holes, threshold = 0.10**2)
                #raw_cd_minus_holes = np.sqrt(np.asarray(raw_cd_minus_holes.cpu()[0]))
                #raw_cd_back_minus_holes = np.sqrt(np.asarray(raw_cd_back_minus_holes.cpu()[0]))
                #cd_minus_holes = raw_cd_minus_holes.mean() + raw_cd_back_minus_holes.mean()

                cds.append(cd)
                f_at_5cms.append(f_at_5cm.cpu().numpy())
                f_at_10cms.append(f_at_10cm.cpu().numpy())
                #cds_minus_holes.append(cd_minus_holes)
                #f_at_5cms_minus_holes.append(f_at_5cm_minus_holes.cpu().numpy())
                #f_at_10cms_minus_holes.append(f_at_10cm_minus_holes.cpu().numpy())
          
                # multi-obj case
                #for obj_index, sil_obj in enumerate(sil_objs):
                for obj_index, sil_gt_obj in enumerate(sil_gt_objs):
                    # sil_gt_obj.shape = (H, W)
                    
                    # MSE (units: m^2)
                    
                    # multi-obj case
                    #mask_obj = (sil_obj > 0.5)  
                    mask_obj = (sil_gt_obj == 1)        # shape = (H, W)                                                     # shape = (H, W)
                    mask_obj_rgb = np.repeat(mask_obj[..., np.newaxis], 3, axis=-1)                         # shape = (H, W, 3)
                    mask_obj_0 = torch.Tensor(mask_obj_rgb[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))    # shape = (1, 3, H, W)
                    rgb_error_obj = np.mean(np.power(rgb_diff[mask_obj_rgb], 2))
                    
                    #depth_error_obj = np.mean(np.power(depth_diff[mask_obj], 2))
                    depth_error_obj = compute_depth_error(dph_gt, dph, conf_gt, mask = mask_obj, dep_scale=opts_list[-1].dep_scale)
                    
                    rgb_errors_objs[obj_index].append(rgb_error_obj)
                    depth_errors_objs[obj_index].append(depth_error_obj)

                    # PSNR
                    #psnr_obj = calculate_psnr(rgb_gt, rgb, mask_obj_rgb)
                    psnr_obj = compute_psnr(rgb_gt, rgb, mask = mask_obj)
                    
                    psnrs_objs[obj_index].append(psnr_obj)

                    # SSIM
                    #ssim_obj = calculate_ssim(rgb_gt, rgb, mask_obj_rgb)
                    ssim_obj = compute_ssim(rgb_gt, rgb, mask = mask_obj)
                    
                    ssims_objs[obj_index].append(ssim_obj)
                    
                    # LPIPS
                    #lpips_obj = lpips_model.forward(rgb_gt_0, rgb_0, mask_obj_0).item()
                    lpips_obj = compute_lpips(rgb_gt, rgb, lpips_model, mask = mask_obj)

                    lpipss_objs[obj_index].append(lpips_obj)

                    # chamfer dist and F-score
                    '''
                    # filter out for points with low confidence
                    x_coord_valid_obj = x_coord[(conf_gt > 1.5) & mask_obj]                          # (H, W)[(H, W) & (H, W)]                                                             
                    y_coord_valid_obj = y_coord[(conf_gt > 1.5) & mask_obj]                          # (H, W)[(H, W) & (H, W)]                                                             
                    P_gt_valid_obj = P_gt[y_coord_valid_obj, x_coord_valid_obj, :]                   # (N_valid_obj, 3)
                    P_valid_obj = P[y_coord_valid_obj, x_coord_valid_obj, :]                         # (N_valid_obj, 3)

                    # compute metrics
                    raw_cd_obj, raw_cd_back_obj, _, _ = chamLoss(P_gt_valid_obj[None, ...], P_valid_obj[None, ...])
                    f_at_5cm_obj, _, _ = fscore.fscore(raw_cd_obj, raw_cd_back_obj, threshold = 0.05**2)
                    f_at_10cm_obj, _, _ = fscore.fscore(raw_cd_obj, raw_cd_back_obj, threshold = 0.10**2)

                    raw_cd_obj = np.sqrt(np.asarray(raw_cd_obj.cpu()[0]))
                    raw_cd_back_obj = np.sqrt(np.asarray(raw_cd_back_obj.cpu()[0]))
                    cd_obj = raw_cd_obj.mean() + raw_cd_back_obj.mean()
                    '''

                    cd_obj, f_at_5cm_obj, f_at_10cm_obj = compute_chamfer_dist_fscore(dph_gt, dph, conf_gt, K, chamLoss, fscore.fscore, mask=mask_obj, dep_scale=opts_list[-1].dep_scale)

                    cds_objs[obj_index].append(cd_obj)
                    f_at_5cms_objs[obj_index].append(f_at_5cm_obj.cpu().numpy())
                    f_at_10cms_objs[obj_index].append(f_at_10cm_obj.cpu().numpy())

        rgbs_gt_refcam.append(rgb_gt_refcam)
        dph_gt_refcam = depth_to_image(torch.from_numpy(dph_gt_refcam[..., None]))
        dphs_gt_refcam.append(dph_gt_refcam*255)
        confs_gt_refcam.append(conf_gt_refcam)
        
        # multi-obj case
        #sils_gt_refcam.append(sil_gt_refcam*255)
        for obj_index, sil_gt_refcam in enumerate(sil_gt_refcam_objs):
            # let's just plot the mask of the fg in question and frames where all pixels == 255
            # i.e. let's ignore entries == 254, which denote the pixels belonging to "other object"
            sil_gt_refcam_filtered = sil_gt_refcam.copy()
            sil_gt_refcam_filtered[sil_gt_refcam_filtered == 254] = 0
            sil_gt_refcam_filtered = np.where(sil_gt_refcam_filtered == 255 , sil_gt_refcam_filtered, sil_gt_refcam_filtered * 255)

            sils_gt_refcam_objs[obj_index].append(sil_gt_refcam_filtered)
            cv2.imwrite('%s-silgt-obj%d_%05d.png'%(opts.nvs_outpath, obj_index, frame_index), sil_gt_refcam_filtered)

        rgbs.append(rgb)
        dph = depth_to_image(torch.from_numpy(dph[..., None]))
        dphs.append(dph*255)
        sils.append(sil*255)
        viss.append(vis*255)

        if opts.stereo_view:
            rgbs_gt_secondcam.append(rgb_gt_secondcam)
            dph_gt_secondcam = depth_to_image(torch.from_numpy(dph_gt_secondcam[..., None]))
            dphs_gt_secondcam.append(dph_gt_secondcam*255)
            confs_gt_secondcam.append(conf_gt_secondcam)

            # multi-obj case
            #sils_gt_secondcam.append(sil_gt_secondcam*255)
            for obj_index, sil_gt_secondcam in enumerate(sil_gt_secondcam_objs):
                # let's just plot the mask of the fg in question and frames where all pixels == 255
                # i.e. let's ignore entries == 254, which denote the pixels belonging to "other object"
                sil_gt_secondcam_filtered = sil_gt_secondcam.copy()
                sil_gt_secondcam_filtered[sil_gt_secondcam_filtered == 254] = 0
                sil_gt_secondcam_filtered = np.where(sil_gt_secondcam_filtered == 255 , sil_gt_secondcam_filtered, sil_gt_secondcam_filtered * 255)

                sils_gt_secondcam_objs[obj_index].append(sil_gt_secondcam_filtered)
                cv2.imwrite('%s-silgt-obj%d_%05d.png'%(opts.nvs_outpath, obj_index, frame_index), sil_gt_secondcam_filtered)

        for obj_index, sils_obj in sils_objs.items():
            sils_obj.append(sil_objs[obj_index]*255)
            cv2.imwrite('%s-sil-obj%d_%05d.png'%(opts.nvs_outpath,obj_index,frame_index), sil_objs[obj_index]*255)    

        cv2.imwrite('%s-rgbgt_%05d.png'%(opts.nvs_outpath,frame_index), rgb_gt_refcam[...,::-1]*255)
        cv2.imwrite('%s-dphgt_%05d.png'%(opts.nvs_outpath,frame_index), dph_gt_refcam[...,::-1]*255)
        #cv2.imwrite('%s-silgt_%05d.png'%(opts.nvs_outpath,i), sil_gt_refcam*255)    

        if opts.stereo_view:
            cv2.imwrite('%s-rgbgt_secondcam_%05d.png'%(opts.nvs_outpath,frame_index), rgb_gt_secondcam[...,::-1]*255)
            cv2.imwrite('%s-dphgt_secondcam_%05d.png'%(opts.nvs_outpath,frame_index), dph_gt_secondcam[...,::-1]*255)
            # multi-obj case
            #cv2.imwrite('%s-silgt_secondcam_%05d.png'%(opts.nvs_outpath,i), sil_gt_secondcam*255)

        if opts.filter_3d:
            cv2.imwrite('%s-rgb_with_asset_%05d.png'%(opts.nvs_outpath,frame_index), rgb_with_asset[...,::-1]*255)

        #* sil_objs[0][..., None] + 255 * (1 - sil_objs[0][..., None]
        cv2.imwrite('%s-rgb_%05d.png'%(opts.nvs_outpath,frame_index), rgb[...,::-1]*255)
        cv2.imwrite('%s-dph_%05d.png'%(opts.nvs_outpath,frame_index), dph[...,::-1]*255)
        cv2.imwrite('%s-sil_%05d.png'%(opts.nvs_outpath,frame_index), sil*255)
        cv2.imwrite('%s-vis_%05d.png'%(opts.nvs_outpath,frame_index), vis*255)
    
    ###################################################################
    ################### modified by Chonghyuk Song ####################
    #save_vid('%s-rgb'%(opts.nvs_outpath), rgbs, suffix='.mp4')
    #save_vid('%s-sil'%(opts.nvs_outpath), sils, suffix='.mp4')
    #save_vid('%s-vis'%(opts.nvs_outpath), viss, suffix='.mp4')

    # scale dphs (automatically done now in save_vid)
    #dphs_total = np.stack(dphs, axis = 0)
    #dphs_max = dphs_total.max()
    #print("255 / dphs_max: {}".format(255 / dphs_max))
    #dphs = [255 / dphs_max * dph for dph in dphs]

    # save errors
    if opts.evaluate:
        if opts.input_view or opts.stereo_view:
            # errors for all frames
            np.save("%s-rgberrors.npy"%(opts.nvs_outpath), np.stack(rgb_errors))
            np.save("%s-deptherrors.npy"%(opts.nvs_outpath), np.stack(depth_errors))
            #np.save("%s-rgberrors-minusholes.npy"%(opts.nvs_outpath), np.stack(rgb_errors_minus_holes))
            #np.save("%s-deptherrors-minusholes.npy"%(opts.nvs_outpath), np.stack(depth_errors_minus_holes))
            
            np.save("%s-psnrs.npy"%(opts.nvs_outpath), np.stack(psnrs))
            np.save("%s-ssims.npy"%(opts.nvs_outpath), np.stack(ssims))
            np.save("%s-lpipss.npy"%(opts.nvs_outpath), np.stack(lpipss))

            # metrics averaged over all frames
            #np.save("%s-rmsrgberror.npy"%(opts.nvs_outpath), np.sqrt(np.mean(np.stack(rgb_errors))))
            #np.save("%s-rmsdeptherror.npy"%(opts.nvs_outpath), np.sqrt(np.mean(np.stack(depth_errors))))
            np.save("%s-rmsrgberror.npy"%(opts.nvs_outpath), rms_metric_over_allframes(rgb_errors))
            np.save("%s-rmsdeptherror.npy"%(opts.nvs_outpath), rms_metric_over_allframes(depth_errors))
            #np.save("%s-rmsrgberror-minusholes.npy"%(opts.nvs_outpath), np.sqrt(np.mean(np.stack(rgb_errors_minus_holes))))
            #np.save("%s-rmsdeptherror-minusholes.npy"%(opts.nvs_outpath), np.sqrt(np.mean(np.stack(depth_errors_minus_holes))))

            np.save("%s-psnr.npy"%(opts.nvs_outpath), average_metric_over_allframes(psnrs))
            np.save("%s-ssim.npy"%(opts.nvs_outpath), average_metric_over_allframes(ssims))
            np.save("%s-lpips.npy"%(opts.nvs_outpath), average_metric_over_allframes(lpipss))
            np.save("%s-cd.npy"%(opts.nvs_outpath), average_metric_over_allframes(cds))
            np.save("%s-fat5cm.npy"%(opts.nvs_outpath), average_metric_over_allframes(f_at_5cms))
            np.save("%s-fat10cm.npy"%(opts.nvs_outpath), average_metric_over_allframes(f_at_10cms))

            #np.save("%s-cd-minusholes.npy"%(opts.nvs_outpath), np.mean(np.stack(cds_minus_holes)))
            #np.save("%s-fat5cm-minusholes.npy"%(opts.nvs_outpath), np.mean(np.stack(f_at_5cms_minus_holes)))
            #np.save("%s-fat10cm-minusholes.npy"%(opts.nvs_outpath), np.mean(np.stack(f_at_10cms_minus_holes)))

            rgb_error_maps_max = np.max(np.stack(rgb_error_maps))
            depth_error_maps_max = np.max(np.stack(depth_error_maps))

            save_vid('%s-rgbabsdiff'%(opts.nvs_outpath), [error_to_image(np.sqrt(rgb_error_map) / np.sqrt(rgb_error_maps_max)) for rgb_error_map in rgb_error_maps], suffix='.mp4', upsample_frame=-1, fps=10)
            
            if opts.stereo_view:
                save_vid('%s-dphabsdiff'%(opts.nvs_outpath), [error_to_image(np.sqrt(depth_error_map) / np.sqrt(depth_error_maps_max), conf_gt_secondcam) for depth_error_map, conf_gt_secondcam in zip(depth_error_maps, confs_gt_secondcam)], suffix='.mp4', upsample_frame=-1, fps=10)
            if opts.input_view:
                save_vid('%s-dphabsdiff'%(opts.nvs_outpath), [error_to_image(np.sqrt(depth_error_map) / np.sqrt(depth_error_maps_max), conf_gt) for depth_error_map, conf_gt in zip(depth_error_maps, confs_gt_refcam)], suffix='.mp4', upsample_frame=-1, fps=10)

            print("[ENTIRE IMAGE] RMS RGB ERROR for {} = {}".format(opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(rgb_errors)))))
            #print("[ENTIRE IMAGE MINUS HOLES] RMS RGB ERROR for {} = {}".format(opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(rgb_errors_minus_holes)[nonempty_frames]))))
            print("[ENTIRE IMAGE] PSNR: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(psnrs))))
            print("[ENTIRE IMAGE] SSIM: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(ssims))))
            print("[ENTIRE IMAGE] LPIPS: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(lpipss))))
            print("[ENTIRE IMAGE] RMS DEPTH ERROR for {} = {}".format(opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(depth_errors)))))
            #print("[ENTIRE IMAGE MINUS HOLES] RMS DEPTH ERROR for {} = {}".format(opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(depth_errors_minus_holes)[nonempty_frames]))))
            print("[ENTIRE IMAGE] CD: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(cds))))
            #print("[ENTIRE IMAGE MINUS HOLES] CD: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(cds_minus_holes))))
            print("[ENTIRE IMAGE] F @ 5cm: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_5cms))))
            #print("[ENTIRE IMAGE MINUS HOLES] F @ 5cm: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_5cms_minus_holes))))
            print("[ENTIRE IMAGE] F @ 10cm: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_10cms))))
            #print("[ENTIRE IMAGE MINUS HOLES] F @ 10cm: {} = {}".format(opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_10cms_minus_holes))))

            for obj_index in range(len(opts_list)):
                rgb_errors_obj = rgb_errors_objs[obj_index]
                depth_errors_obj = depth_errors_objs[obj_index]
                psnrs_obj = psnrs_objs[obj_index]
                ssims_obj = ssims_objs[obj_index]
                lpipss_obj = lpipss_objs[obj_index]
                cds_obj = cds_objs[obj_index]
                f_at_5cms_obj = f_at_5cms_objs[obj_index]
                f_at_10cms_obj = f_at_10cms_objs[obj_index]

                # fixing case where (sil_gt_obj == 1) is all False (when scene is entirely fg or bkgd)
                nonempty_frames_obj = np.logical_not(np.isnan(np.stack(rgb_errors_obj)))            # frames with valid masks

                # errors for all frames
                np.save("%s-rgberrors-obj%d.npy"%(opts.nvs_outpath, obj_index), np.stack(rgb_errors_obj))
                np.save("%s-deptherrors-obj%d.npy"%(opts.nvs_outpath, obj_index), np.stack(depth_errors_obj))
                
                # errors averaged over all frames
                np.save("%s-rmsrgberror-obj%d.npy"%(opts.nvs_outpath, obj_index), rms_metric_over_allframes(rgb_errors_obj))
                np.save("%s-rmsdeptherror-obj%d.npy"%(opts.nvs_outpath, obj_index), rms_metric_over_allframes(depth_errors_obj))

                # perceptual metrics averaged over all frames
                np.save("%s-psnr-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(psnrs_obj))
                np.save("%s-ssim-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(ssims_obj))
                np.save("%s-lpips-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(lpipss_obj))
                np.save("%s-cd-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(cds_obj))
                np.save("%s-fat5cm-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(f_at_5cms_obj))
                np.save("%s-fat10cm-obj%d.npy"%(opts.nvs_outpath, obj_index), average_metric_over_allframes(f_at_10cms_obj))

                print("[OBJ {}] RMS RGB ERROR for {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(rgb_errors_obj)[nonempty_frames_obj]))))
                print("[OBJ {}] PSNR: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(psnrs_obj)[nonempty_frames_obj])))
                print("[OBJ {}] SSIM: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(ssims_obj)[nonempty_frames_obj])))
                print("[OBJ {}] LPIPS: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(lpipss_obj)[nonempty_frames_obj])))
                print("[OBJ {}] RMS DEPTH ERROR for {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.sqrt(np.mean(np.stack(depth_errors_obj)[nonempty_frames_obj]))))
                print("[OBJ {}] CD: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(cds_obj)[nonempty_frames_obj])))
                print("[OBJ {}] F @ 5cm: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_5cms_obj)[nonempty_frames_obj])))
                print("[OBJ {}] F @ 10cm: {} = {}".format(obj_index, opts.nvs_outpath.split("-")[-1], np.mean(np.stack(f_at_10cms_obj)[nonempty_frames_obj])))

    print("opts.nvs_outpath")

    save_vid('%s-rgbgt'%(opts.nvs_outpath), rgbs_gt_refcam, suffix='.mp4', upsample_frame=-1, fps=10)
    save_vid('%s-dphgt'%(opts.nvs_outpath), dphs_gt_refcam, suffix='.mp4', upsample_frame=-1, fps=10)
    
    # multi-obj case
    #save_vid('%s-silgt'%(opts.nvs_outpath), sils_gt_refcam, suffix='.mp4', upsample_frame=-1, fps=10)
    for obj_index, sils_gt_refcam_obj in sils_gt_refcam_objs.items():
        save_vid('%s-silgt-obj%d'%(opts.nvs_outpath, obj_index), sils_gt_refcam_obj, suffix='.mp4', upsample_frame=-1, fps=10)

    if opts.stereo_view:
        save_vid('%s-rgbgt-secondcam'%(opts.nvs_outpath), rgbs_gt_secondcam, suffix='.mp4', upsample_frame=-1, fps=10)
        save_vid('%s-dphgt-secondcam'%(opts.nvs_outpath), dphs_gt_secondcam, suffix='.mp4', upsample_frame=-1, fps=10)
        
        # multi-obj case
        #save_vid('%s-silgt-secondcam'%(opts.nvs_outpath), sils_gt_secondcam, suffix='.mp4', upsample_frame=-1, fps=10)
        for obj_index, sils_gt_secondcam_obj in sils_gt_secondcam_objs.items():
            save_vid('%s-silgt-obj%d'%(opts.nvs_outpath, obj_index), sils_gt_secondcam_obj, suffix='.mp4', upsample_frame=-1, fps=10)

    if opts.filter_3d:
        save_vid('%s-rgb_with_asset'%(opts.nvs_outpath), rgbs_with_asset, suffix='.mp4', upsample_frame=-1, fps=10)

    save_vid('%s-rgb'%(opts.nvs_outpath), rgbs, suffix='.mp4', upsample_frame=-1, fps=10)
    save_vid('%s-dph'%(opts.nvs_outpath), dphs, suffix='.mp4', upsample_frame=-1, fps=10)
    save_vid('%s-sil'%(opts.nvs_outpath), sils, suffix='.mp4', upsample_frame=-1, fps=10)

    for obj_index, sils_obj in sils_objs.items():
        save_vid('%s-sil-obj%d'%(opts.nvs_outpath, obj_index), sils_obj, suffix='.mp4', upsample_frame=-1, fps=10)

    save_vid('%s-vis'%(opts.nvs_outpath), viss, suffix='.mp4', upsample_frame=-1, fps=10)
    ###################################################################
    ###################################################################

if __name__ == '__main__':
    app.run(main)
