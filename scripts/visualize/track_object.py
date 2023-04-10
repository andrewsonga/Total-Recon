# script to visualize the bkgd mesh as well as the trajectory of the fg in bkgd coordinates

from re import M
from absl import flags, app
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
sys.path.insert(0,'')
sys.path.insert(0,'third_party')
import subprocess
import imageio
import glob
from utils.io import save_vid
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pdb
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, obj2cam_np
from dataloader import frameloader
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer
import configparser
import matplotlib
cmap = matplotlib.cm.get_cmap('cool')
from utils.io import config_to_dataloader, draw_cams, str_to_frame, \
        extract_data_info, load_root
import pytorch3d
import pytorch3d.ops
import copy

flags.DEFINE_string('seqname', '', 'name of the sequence')
flags.DEFINE_string('rootdir', '', 'directory of the camera poses')
flags.DEFINE_string('nvs_outpath', 'tmp/nvs-','output prefix')
flags.DEFINE_integer('startframe', 0, 'number of start frame to render')
flags.DEFINE_integer('maxframe', -1, 'maximum number frame to render')
flags.DEFINE_integer('fix_frame', -1, 'frame number to fix camera at: -1 denotes user-defined camera view')
flags.DEFINE_float('topdowncam_offset_y', 0.24, 'offset along the y-direction of the fgroot frame for topdown view')
flags.DEFINE_float('topdowncam_offset_x', 0., 'offset along the x-direction of the fgroot frame for topdown view')
flags.DEFINE_float('topdowncam_offset_z', 0., 'offset along the z-direction of the fgroot frame for topdown view')
flags.DEFINE_float('scale', 0.1,
        'scale applied to the rendered image (wrt focal length)')
opts = flags.FLAGS

def main(_):
    loadname_objs = ["logdir/{}/obj0/".format(opts.seqname), "logdir/{}/obj1/".format(opts.seqname)]
    rootdir_objs = ["logdir/{}/obj0/".format(opts.seqname), "logdir/{}/obj1/".format(opts.seqname)]

    opts_list = []

    for loadname, rootdir in zip(loadname_objs, rootdir_objs):
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.seqname = opts.seqname
        opts_obj.rootdir = rootdir
        opts_list.append(opts_obj)

    ##################################################
    # 1. load rest mest for background and foreground
    ##################################################
    #mesh_rest_bkgd = trimesh.load('%s/mesh_rest-14.obj'%(rootdir_objs[-1]),process=False)
    #mesh_rest_fg = trimesh.load('%s/mesh_rest-14.obj'%(rootdir_objs[0]),process=False)
    mesh_rest_bkgd = trimesh.load('%s/mesh-rest.obj'%(rootdir_objs[-1]),process=False)
    mesh_rest_bkgd.visual.vertex_colors[:, :3] = 128                    # color = gray 
    mesh_rest_fg = trimesh.load('%s/mesh-rest.obj'%(rootdir_objs[0]),process=False)

    ###########################################
    # 2. load the camera poses for each object
    ###########################################
    rtks_objs = []
    for opts_obj in opts_list:
        rtks = load_root(opts_obj.rootdir, 0)  # cap frame=0=>load all
        rtks_objs.append(rtks)

    for rtks in rtks_objs:
        rtks[:,3] = rtks[:,3]*opts.scale             # scaling intrinsics

    ###########################################
    # 3. determine image height and width etc.
    ###########################################
    if opts.maxframe > 0:
        size = opts.maxframe - opts.startframe
    else:
        size = len(rtks_objs[-1]) - 1 - opts.startframe

    # hardcoded for now
    rndsils = np.ones((size, 960, 720))             # (N, H, W)        # set opts.scale to 0.375
    img_size = rndsils[0].shape
    if img_size[0] > img_size[1]:
        img_type='vert'
    else:
        img_type='hori'

    img_size = int(max(img_size)*opts.scale)

    if img_type=='vert':
        size_short_edge = int(rndsils[0].shape[1] * img_size/rndsils[0].shape[0])
        height = img_size
        width = size_short_edge        
    else:
        size_short_edge = int(rndsils[0].shape[0] * img_size/rndsils[0].shape[1])
        height = size_short_edge
        width = img_size

    print("height: {}".format(height))
    print("width: {}".format(width))

    ########################################################
    # 4. compute the bkgd2novelcams transformation matrices
    ########################################################
    if opts.maxframe > 0:
        sample_idx = np.arange(opts.startframe,opts.maxframe).astype(int)
    else:
        sample_idx = np.arange(opts.startframe,len(rtks)-1).astype(int)

    if opts.fix_frame >= 0:
        # camera view fixed to a specific frame's view
        rtks_objs_fixframe = [rtks[np.repeat(opts.fix_frame, size)] for rtks in rtks_objs]          # list of object poses of shape (size, 4, 4) where the poses are same for every row - to be used as the fixed pose in bkgd space from which to view the meshes (obj pose, camera pose, and bkgd)
        rtks_objs = [rtks[sample_idx] for rtks in rtks_objs]                                        # list of object poses of shape (size, 4 ,4) where each row represents the pose at the corresponding frame idx
        
        # topdown view corresponding to the camera view of "fix_frame"
        
        # assuming that the fgmeshcenter frame is well aligned with the fg object itself, we can define an offset along the y axis of the root frame of the fg and view the fg object from there
        # in the case of the rest mesh of the cat in the video cat-pikachiu-rgbd000, this offset should be a few units in the y-direction
        bkgd2cam = np.concatenate((rtks_objs_fixframe[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fixframe[-1].shape[0], axis = 0)), axis = 1)
        fg2cam = np.concatenate((rtks_objs_fixframe[0][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs_fixframe[0].shape[0], axis = 0)), axis = 1)
        cam2bkgd = np.linalg.inv(bkgd2cam)
        fg2bkgd = np.matmul(cam2bkgd, fg2cam)                                                       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates                  
        
        topdowncam2fg = np.eye(4)
        topdowncam2fg[:3, :3] = cv2.Rodrigues(np.asarray([np.pi/2., 0., 0.]))[0]
        
        topdowncam2fg[0, 3] = opts.topdowncam_offset_x                                                # offset along the x-direction of the fgroot frame for topdown view
        topdowncam2fg[1, 3] = opts.topdowncam_offset_y                                                # offset along the y-direction of the fgroot frame for topdown view
        topdowncam2fg[2, 3] = opts.topdowncam_offset_z                                                # offset along the z-direction of the fgroot frame for topdown view
        
        elevatedfg2bkgd = np.matmul(fg2bkgd, np.repeat(topdowncam2fg[None, ...], fg2bkgd.shape[0], axis = 0))
        bkgd2novelcams = np.linalg.inv(elevatedfg2bkgd)                                             # (N, 4, 4)
        bkgd2novelcams = torch.Tensor(bkgd2novelcams).cuda()

    ###########################################
    # 5. load the vertices for background mesh
    ###########################################
    Rmat = bkgd2novelcams[0:1, :3, :3]                                                  # shape = (1, 3, 3)
    Tmat = bkgd2novelcams[0:1, :3, 3]                                                   # shape = (1, 3)
    focal = rtks_objs[-1][0, 3, :2]                                         
    ppoint = rtks_objs[-1][0, 3, 2:]
    mesh_rest_bkgd.visual.vertex_colors[:, :3] = 64
    colors = mesh_rest_bkgd.visual.vertex_colors

    refface_bkgd = torch.Tensor(mesh_rest_bkgd.faces[None]).cuda()
    verts_bkgd = torch.Tensor(mesh_rest_bkgd.vertices[None]).cuda()
    verts_bkgd = obj_to_cam(verts_bkgd, Rmat, Tmat)                                     # need to input Rmat of shape (1, 3, 3) and Tmat of shape (1,3), where Rmat, Tmat denote obj2cam matrices (extrinsics)
    mesh_rest_bkgd = trimesh.Trimesh(vertices=np.asarray(verts_bkgd[0,:,:3].cpu()), faces=np.asarray(refface_bkgd[0].cpu()), vertex_colors=colors)
    meshr_rest_bkgd = Mesh.from_trimesh(mesh_rest_bkgd,smooth=True)
    meshr_rest_bkgd._primitives[0].material.RoughnessFactor=1.

    mesh_rnd_colors = []
    for i, frame_idx in enumerate(sample_idx):
        print("rendering object and camera poses: index {}".format(i))
        #################################################################
        # 6. draw_cams with the bkgd2fg transformation matrices as input
        #################################################################
        assert(len(rtks_objs) == 2)                                                                     # check if there are two, and only two objects that we are loading
        #bkgd2cam = np.concatenate((rtks_objs[-1][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[-1].shape[0], axis = 0)), axis = 1)
        #fg2cam = np.concatenate((rtks_objs[0][:, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), rtks_objs[0].shape[0], axis = 0)), axis = 1)
        bkgd2cam = np.concatenate((rtks_objs[-1][0:i+1, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), i+1, axis = 0)), axis = 1)
        fg2cam = np.concatenate((rtks_objs[0][0:i+1, :3, :], np.repeat(np.array([[[0., 0., 0., 1.]]]), i+1, axis = 0)), axis = 1)
        cam2fg = np.linalg.inv(fg2cam)
        #cam2fg[:, :3, 3] -= np.repeat(mesh_rest_fg.vertices.mean(0)[None, :], cam2fg.shape[0], axis = 0)     # shifting the translation component of cam2fg to account for center of the fg rest mest
        bkgd2fg = np.matmul(cam2fg, bkgd2cam)                               # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)
        bkgd2fg = [bkgd2fg_i for bkgd2fg_i in bkgd2fg]                      # a list of N (4,4) matrices, each representing the pose of the fg root body in the bkgd frame
        mesh_fgroot = draw_cams(bkgd2fg, color = 'hot')                    # contains a mesh with vertices defined in the bkgd frame
        mesh_cams = draw_cams(bkgd2cam, color = 'cool')

        ################################################################################################
        # 7. project the mesh vertices for each mesh in the scene (background, obj poses, camera poses)
        # from the bkgd space to the desired camera space with bkgd2novelcams
        ################################################################################################
        mesh_fgroot_transformed = mesh_fgroot.copy()
        mesh_cams_transformed = mesh_cams.copy()
        
        mesh_fgroot_transformed.vertices = obj2cam_np(mesh_fgroot_transformed.vertices, Rmat, Tmat)
        mesh_cams_transformed.vertices = obj2cam_np(mesh_cams_transformed.vertices, Rmat, Tmat)
        #mesh_fgroot_transformed = mesh_fgroot.copy()
        #mesh_fgroot_transformed.export('logdir/%s/mesh_fgroot.obj'%(opts.seqname))
        #mesh_cams.export('logdir/%s/mesh_cams.obj'%(opts.seqname))

        #################################################################
        # 8. create a OffscreenRenderer object and Scene object
        #################################################################
        r = OffscreenRenderer(img_size, img_size)
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
        colors= np.concatenate([0.6*colors[:,:3].astype(np.uint8), colors[:,3:]],-1)        # avoid overexposure

        ###############################################################################
        # 9. add all of the meshes (obj pose, camera pose, bkgd) to the Scene as nodes
        ###############################################################################
        meshr_fgroot_transformed=Mesh.from_trimesh(mesh_fgroot_transformed, smooth=False)
        meshr_fgroot_transformed._primitives[0].material.RoughnessFactor=.5
        scene.add_node( Node(mesh=meshr_fgroot_transformed))

        meshr_cams_transformed=Mesh.from_trimesh(mesh_cams_transformed, smooth=False)
        meshr_cams_transformed._primitives[0].material.RoughnessFactor=.5
        scene.add_node( Node(mesh=meshr_cams_transformed))
        scene.add_node( Node(mesh=meshr_rest_bkgd))

        '''
        bkgd2cam_fixframe = bkgd2cam[opts.fix_frame, ...]
        fg2cam_fixframe = fg2cam[opts.fix_frame, ...]
        cam2bkgd_fixframe = np.linalg.inv(bkgd2cam_fixframe)
        fg2bkgd_fixframe = np.matmul(cam2bkgd_fixframe, fg2cam_fixframe)       # (N, 4, 4) x (N, 4, 4) -> (N, 4, 4)    # the R and t of fg2bkgd represent the orientation and position of fg root frame in bkgd coordinates                  
        topdowncam2fg = np.eye(4)
        topdowncam2fg[:3, :3] = cv2.Rodrigues(np.asarray([np.pi/2. +np.pi/3 , 0., 0.]))[0]
        topdowncam2fg[1, 3] = opts.topdowncam_offset
        elevatedfg2bkgd_fixframe = np.matmul(fg2bkgd_fixframe, topdowncam2fg)
        cam_pose = np.linalg.inv(elevatedfg2bkgd_fixframe)      # cam poses should be bkgd2cam transformations in OpenGL format, meaning we have to multiply 1st and 2nd row by -1
        cam_pose[1, :] = -1 * cam_pose[1, :]
        cam_pose[2, :] = -1 * cam_pose[2, :]
        '''

        cam = IntrinsicsCamera(
                focal[0],
                focal[0],
                ppoint[0],
                ppoint[1],
                znear=1e-3,zfar=1000)
        cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
        scene.add(cam, pose=cam_pose)            # pose w.r.t. world frame (i.e. pose of the camera in bkgd frame i.e. cam2bkgd transformation)

        #####################################
        # 10. add a light source to the scene
        #####################################
        theta = np.pi
        phi = -1/12 * np.pi
        init_light_pose = np.matmul(np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]]), np.asarray([[np.cos(phi),0,-np.sin(phi),0], [0,1,0,0],[np.sin(phi),0,np.cos(phi),0],[0,0,0,1]]))
        light_pose = init_light_pose
        direc_l_node = scene.add(direc_l, pose=light_pose)

        #########################################################################################
        # 11. render the scene from the camera view of the first frame (bkgd2cam of first frame)
        #########################################################################################
        mesh_rnd_color, mesh_rnd_depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()

        mesh_rnd_color = mesh_rnd_color[:height, :width, :3]
        mesh_rnd_colors.append(mesh_rnd_color)
        cv2.imwrite('%s_%05d.png'%(opts.nvs_outpath,i), mesh_rnd_color[:,:,::-1])

    save_vid('%s'%(opts.nvs_outpath), mesh_rnd_colors, suffix='.mp4', upsample_frame=-1, fps=10)

if __name__ == '__main__':
    app.run(main)