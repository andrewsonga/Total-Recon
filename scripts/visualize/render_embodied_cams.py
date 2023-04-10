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
flags.DEFINE_float('firstpersoncam_offset_x', 0, 'offset along the y-direction of the fgmeshcenter frame for first-person view')
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
flags.DEFINE_bool('render_cam_inputview', False, 'render the 3d asset of the camera from the inputview')
flags.DEFINE_bool('render_cam_stereoview', False, 'render the 3d asset of the camera from the stereoview')
flags.DEFINE_bool('render_cam_fixedview', False, 'render the 3d asset of the camera from the fixedview')
flags.DEFINE_bool('input_view', False, 'render from the camera trajectory from the original input video')
flags.DEFINE_bool('evaluate', True, 'computing evaluation metrics')
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


    opts.nvs_outpath = opts.nvs_outpath + "-povcams"

    rtks_objs_fps = [rtks.copy() for rtks in rtks_objs]

    if opts.maxframe > 0:
        #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
        sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
    else:
        #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
        sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
    rtks_objs_fps = [rtks[sample_idx] for rtks in rtks_objs_fps]

    # what should the novel-camera trajectory (w.r.t bkgd frame) be?
    # 
    # assuming that the fgmeshcenter frame is well aligned with the cat, we can set an offset along the xz plane of the fgmeshcenter frame and view the cat from there
    # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should a few units behind the cat in the z-direction
    bkgd2cam = np.concatenate((rtks_objs_fps[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fps[-1].shape[0], axis = 0)), axis = 1)
    fg2cam = np.concatenate((rtks_objs_fps[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fps[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
    cam2bkgd = np.linalg.inv(bkgd2cam)
    fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates  

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
    bkgd2videocams = np.copy(rtks_objs_fps[-1])      # (N, 4, 4)
    bkgd2videocams[:, 3, :] = 0.
    bkgd2videocams[:, 3, 3] = 1.

    for rtks_obj in rtks_objs_fps:
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

    if opts.maxframe > 0:
        #sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
        sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
    else:
        #sample_idx= np.linspace(0,len(rtks)-1,len(rtks)).astype(int)
        sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)
    rtks_objs_tps = [rtks[sample_idx] for rtks in rtks_objs]

    # what should the novel-camera trajectory (w.r.t bkgd frame) be?
    # 
    # assuming that the fgmeshcenter frame is well aligned with the cat, we can set an offset along the xz plane of the fgmeshcenter frame and view the cat from there
    # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should a few units behind the cat in the z-direction
    bkgd2cam = np.concatenate((rtks_objs_tps[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_tps[-1].shape[0], axis = 0)), axis = 1)
    fg2cam = np.concatenate((rtks_objs_tps[opts.fg_obj_index][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_tps[opts.fg_obj_index].shape[0], axis = 0)), axis = 1)
    cam2bkgd = np.linalg.inv(bkgd2cam)
    fg2bkgd = np.matmul(cam2bkgd, fg2cam)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates  

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
    bkgd2videocams = np.copy(rtks_objs_tps[-1])      # (N, 4, 4)
    bkgd2videocams[:, 3, :] = 0.
    bkgd2videocams[:, 3, 3] = 1.

    for rtks_obj in rtks_objs_tps:
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

    ####################################################################################################
    ####################################################################################################
    for rtks in rtks_objs_fps:
        rtks[:,3] = rtks[:,3]*opts.scale             # scaling intrinsics for fps cameras

    for rtks in rtks_objs_tps:
        rtks[:,3] = rtks[:,3]*opts.scale             # scaling intrinsics for tps cameras

    for rtks in rtks_objs_rendercam:
        rtks[:,3] = rtks[:,3]*opts.scale        # scaling intrinsics

    bs = len(rtks_objs_rendercam[-1])
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
    rtks_objs_rendercam_torch = [torch.Tensor(rtks_obj_rendercam).cuda() for rtks_obj_rendercam in rtks_objs_rendercam]

    if opts.filter_3d or True:
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
                               [0,  np.cos(-np.pi / 2), np.sin(-np.pi / 2)],
                               [0, -np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
            mesh_asset.vertices = np.matmul(rotation_abt_xaxis, mesh_asset.vertices.T).T


        mesh_asset_fps = trimesh.load(file_obj='mesh_material/camera.obj', force='mesh')
        mesh_asset_tps = trimesh.load(file_obj='mesh_material/camera.obj', force='mesh')

        # (first-person view camera mesh): color yellow
        color_fps = mesh_asset_fps.visual.to_color()
        mesh_asset_fps.visual.vertex_colors = color_fps
        color_tps = mesh_asset_tps.visual.to_color()
        mesh_asset_tps.visual.vertex_colors = color_tps
            
        mesh_asset_fps.visual.vertex_colors.vertex_colors = np.tile(np.array([[0.5, 0.5, 0, 1.]]), (mesh_asset_fps.visual.vertex_colors.vertex_colors.shape[0], 1))
        # scale vertices of 3d asset
        mesh_asset_fps.vertices = mesh_asset_fps.vertices * opts.asset_scale

        # (third-person view camera mesh): color blue
        mesh_asset_tps.visual.vertex_colors.vertex_colors = np.tile(np.array([[0, 0, 0.5, 1.]]), (mesh_asset_tps.visual.vertex_colors.vertex_colors.shape[0], 1))
        # scale vertices of 3d asset
        mesh_asset_tps.vertices = mesh_asset_tps.vertices * opts.asset_scale

        meshasset_rnd_colors = []
        meshasset_rnd_depths = []

    print("RENDERING MESH")
    mesh_rnd_colors = []
    near_far_values = []

    for i, frame_idx in enumerate(sample_idx):    
        print("index {} / {}".format(i, len(sample_idx)))
        r = OffscreenRenderer(img_size, img_size)
        scene = Scene(ambient_light=0.6*np.asarray([1.,1.,1.,1.]))
    
        for obj_index, opts_obj in enumerate(opts_list):
            meshdir_obj = opts_obj.rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
            #rtks_obj = torch.Tensor(rtks_objs[obj_index]).cuda()            # doesn't seem to be used
        
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
            
            Rmat_obj_time = rtks_objs_rendercam_torch[obj_index][i : i+1, :3, :3]
            Tmat_obj_time = rtks_objs_rendercam_torch[obj_index][i : i+1, :3, 3]

            verts_obj_time = obj_to_cam(verts_obj_time, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
            mesh_obj_time = trimesh.Trimesh(vertices=np.asarray(verts_obj_time[0,:,:3].cpu()), faces=np.asarray(faces_obj_time[0].cpu()), vertex_colors=mesh_obj_time.visual.vertex_colors)
            meshr_obj_time = Mesh.from_trimesh(mesh_obj_time, smooth=True)
            meshr_obj_time._primitives[0].material.RoughnessFactor=1.

            # 3. add the resulting mesh, where vertices are no defined in the desired camera frame to a Scene object
            scene.add_node(Node(mesh=meshr_obj_time))

        # 1. load foreground mesh
        meshdir_fg = opts_list[opts.asset_obj_index].rootdir                                  # "logdir/{}/obj0/".format(opts.seqname)
        meshdir_fg_time = glob.glob(meshdir_fg + "*-mesh-%05d.obj"%(frame_idx))
        assert(len(meshdir_fg_time) == 1)
        mesh_fg_time = trimesh.load(meshdir_fg_time[0], process=False)

        # add first-person view camera asset
        # transform vertices of 3d asset to fg frame using "asset2fg" transformation
        asset2fg_fps = np.eye(4)

        # translation component = coordinate of the camera frame origin in the fgframe (novelcam2fg)                
        fg2novelcam_time_fps = rtks_objs_fps[opts.asset_obj_index][i, ...].copy()
        fg2novelcam_time_fps[:, 3] = fg2novelcam_time_fps[:, 3]
        fg2novelcam_time_fps[3, :] = np.array([0, 0, 0, 1]) 
        novelcam2fg_time_fps = np.linalg.inv(fg2novelcam_time_fps)

        asset2fg_fps[:3, 3] = novelcam2fg_time_fps[:3, 3]     # rtks_obj represents obj2novelcam transformation

        # rotation component:
        # assume unicorn horn and camera both have OpenGL convention (where the normal / principal viewing direction points towards -ve z-axis)
        # extracting Z-axis direction of the 3d asset = vertex normal
        asset2fg_fps[:3, 2] = -novelcam2fg_time_fps[:3, 2]       # in OpenGL (which is the convention the camera asset is defined in ), y-axis is in opposite direction to OpenCV format (which is the convention rtks_obj is defined in)

        # extracting X-axis
        asset2fg_fps[:3, 0] = novelcam2fg_time_fps[:3, 0]

        # extracting Y-axis
        asset2fg_fps[:3, 1] = -novelcam2fg_time_fps[:3, 1]       # in OpenGL, y-axis is in opposite direction to OpenCV format
        asset2fg_fps[:3, 3] = asset2fg_fps[:3, 3] - opts.asset_offset_z * opts_list[-1].dep_scale * asset2fg_fps[:3, 2]

        asset_vertices_assetframe_homo_fps = np.concatenate([mesh_asset_fps.vertices, np.ones_like(mesh_asset_fps.vertices[:, 0:1])], axis = -1)        # (N, 3) + (N, 1) = (N, 4)
        asset_vertices_fgframe_homo_fps = np.matmul(asset2fg_fps, asset_vertices_assetframe_homo_fps.T).T

        # 4. change asset_mesh and fg_mesh from fg frame to camera frame
        #    if opts.render_camera, then we change to the camera frame specified by rtks_objs_rendercam_torch
        faces_asset_time_fps = torch.Tensor(mesh_asset_fps.faces[None]).cuda()
        verts_asset_time_fps = torch.Tensor(asset_vertices_fgframe_homo_fps[:, :3][None]).cuda()
        
        Rmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, :3]
        Tmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, 3]

        verts_asset_time_fps = obj_to_cam(verts_asset_time_fps, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
        mesh_asset_time_fps = trimesh.Trimesh(vertices=np.asarray(verts_asset_time_fps[0,:,:3].cpu()), faces=np.asarray(faces_asset_time_fps[0].cpu()), vertex_colors=mesh_asset_fps.visual.vertex_colors.vertex_colors)     #
        meshr_asset_time_fps = Mesh.from_trimesh(mesh_asset_time_fps, smooth=True)

        # 5. add the resulting mesh, where vertices are now defined in the desired camera frame to a Scene object
        scene.add_node(Node(mesh=meshr_asset_time_fps))

        # add third-person view camera asset
        # transform vertices of 3d asset to fg frame using "asset2fg" transformation
        asset2fg_tps = np.eye(4)

        # translation component = coordinate of the camera frame origin in the fgframe (novelcam2fg)                
        fg2novelcam_time_tps = rtks_objs_tps[opts.asset_obj_index][i, ...].copy()
        fg2novelcam_time_tps[:, 3] = fg2novelcam_time_tps[:, 3]
        fg2novelcam_time_tps[3, :] = np.array([0, 0, 0, 1]) 
        novelcam2fg_time_tps = np.linalg.inv(fg2novelcam_time_tps)

        asset2fg_tps[:3, 3] = novelcam2fg_time_tps[:3, 3]     # rtks_obj represents obj2novelcam transformation

        # rotation component:
        # assume unicorn horn and camera both have OpenGL convention (where the normal / principal viewing direction points towards -ve z-axis)
        # extracting Z-axis direction of the 3d asset = vertex normal
        asset2fg_tps[:3, 2] = -novelcam2fg_time_tps[:3, 2]       # in OpenGL (which is the convention the camera asset is defined in ), y-axis is in opposite direction to OpenCV format (which is the convention rtks_obj is defined in)

        # extracting X-axis
        asset2fg_tps[:3, 0] = novelcam2fg_time_tps[:3, 0]

        # extracting Y-axis
        asset2fg_tps[:3, 1] = -novelcam2fg_time_tps[:3, 1]       # in OpenGL, y-axis is in opposite direction to OpenCV format
        asset2fg_tps[:3, 3] = asset2fg_tps[:3, 3] - opts.asset_offset_z * opts_list[-1].dep_scale * asset2fg_tps[:3, 2]

        asset_vertices_assetframe_homo_tps = np.concatenate([mesh_asset_tps.vertices, np.ones_like(mesh_asset_tps.vertices[:, 0:1])], axis = -1)        # (N, 3) + (N, 1) = (N, 4)
        asset_vertices_fgframe_homo_tps = np.matmul(asset2fg_tps, asset_vertices_assetframe_homo_tps.T).T

        # 4. change asset_mesh and fg_mesh from fg frame to camera frame
        #    if opts.render_camera, then we change to the camera frame specified by rtks_objs_rendercam_torch
        faces_asset_time_tps = torch.Tensor(mesh_asset_tps.faces[None]).cuda()
        verts_asset_time_tps = torch.Tensor(asset_vertices_fgframe_homo_tps[:, :3][None]).cuda()
        
        Rmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, :3]
        Tmat_obj_time = rtks_objs_rendercam_torch[opts.asset_obj_index][i : i+1, :3, 3]

        verts_asset_time_tps = obj_to_cam(verts_asset_time_tps, Rmat_obj_time, Tmat_obj_time)           # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices
        mesh_asset_time_tps = trimesh.Trimesh(vertices=np.asarray(verts_asset_time_tps[0,:,:3].cpu()), faces=np.asarray(faces_asset_time_tps[0].cpu()), vertex_colors=mesh_asset_tps.visual.vertex_colors.vertex_colors)     #
        meshr_asset_time_tps = Mesh.from_trimesh(mesh_asset_time_tps, smooth=True)

        # 5. add the resulting mesh, where vertices are now defined in the desired camera frame to a Scene object
        scene.add_node(Node(mesh=meshr_asset_time_tps))

        # 4. add a camera node with an I_4x4 transformation (adjusted for sign conventions)
        focal_time = rtks_objs_rendercam_torch[-1][i, 3, :2]
        ppoint_time = rtks_objs_rendercam_torch[-1][i, 3, 2:]

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
        birdeyelight2fg = np.matmul(birdeyelight2fg, rotation_abt_xaxis)
        birdeyelight2fg = np.matmul(birdeyelight2fg, rotation_abt_yaxis)
        birdeyelight2fg = np.matmul(birdeyelight2fg, rotation_abt_zaxis)

        if opts.fix_frame < 0:
            i_fixframe = opts.fix_frame
        else:
            i_fixframe = np.argmax(sample_idx == opts.fix_frame)
        bkgd2cam_fixframe = np.concatenate([rtks_objs_rendercam[-1][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
        fg2cam_fixframe = np.concatenate([rtks_objs_rendercam[opts.fg_obj_index][i_fixframe, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
        bkgd2cam_time = np.concatenate([rtks_objs_rendercam[-1][i, :3, :], np.array([[0., 0., 0., 1.]])], axis = 0)
        
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
        near_far_values.append(near_far)

        cv2.imwrite('%s-mesh_%05d.png'%(opts.nvs_outpath,frame_idx), mesh_rnd_color_time[...,::-1])
        mesh_rnd_colors.append(mesh_rnd_color_time)

    save_vid('%s-mesh'%(opts.nvs_outpath), mesh_rnd_colors, suffix='.mp4', upsample_frame=-1, fps=10)
    
if __name__ == '__main__':
    app.run(main)
