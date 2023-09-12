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
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio
import copy
import shutil
import time

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer_objs
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo

#flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
flags.DEFINE_multi_string('loadname_objs', 'None', 'name of folder inside \logdir to load into fg banmo object')
flags.DEFINE_multi_string('savename_objs', 'None', 'name of folder inside \logdir to load into fg banmo object')
flags.DEFINE_bool('extract_mesh',True,'whether or not to extract object meshes')
opts = flags.FLAGS
                
def save_output_obj(obj_index, rendered_seq, aux_seq, save_dir_obj, save_flo, extract_mesh):
    #save_dir_obj = '%s/obj%d/'%(save_dir, obj_index)
    length = len(aux_seq['mesh'])
    mesh_rest = aux_seq['mesh_rest']
    try:
        len_max = (mesh_rest.vertices.max(0) - mesh_rest.vertices.min(0)).max()
        mesh_rest.export('%s/mesh-rest.obj'%save_dir_obj)
        if 'mesh_rest_skin' in aux_seq.keys():
            aux_seq['mesh_rest_skin'].export('%s/mesh-rest-skin.obj'%save_dir_obj)
        if 'bone_rest' in aux_seq.keys():
            bone_rest = aux_seq['bone_rest']
            save_bones(bone_rest, len_max, '%s/bone-rest.obj'%save_dir_obj)
    except:
        pass

    flo_gt_vid = []
    flo_p_vid = []
    for i in range(length):
        impath = aux_seq['impath'][i]
        seqname = impath.split('/')[-2]
        save_prefix = '%s/%s'%(save_dir_obj,seqname)
        idx = int(impath.split('/')[-1].split('.')[-2])
        mesh = aux_seq['mesh'][i]
        rtk = aux_seq['rtk'][i]
        
        if extract_mesh:
            try:
                # convert bones to meshes TODO: warp with a function
                if 'bone' in aux_seq.keys() and len(aux_seq['bone'])>0:
                    bones = aux_seq['bone'][i]
                    bone_path = '%s-bone-%05d.obj'%(save_prefix, idx)
                    save_bones(bones, len_max, bone_path)
                mesh.export('%s-mesh-%05d.obj'%(save_prefix, idx))
            except:
                pass
        np.savetxt('%s-cam-%05d.txt'  %(save_prefix, idx), rtk)
        print("obj %d: saved bone, mesh, and cam index %05d"%(obj_index, idx))
    
def transform_shape(mesh,rtk):
    """
    (deprecated): absorb rt into mesh vertices, 
    """
    vertices = torch.Tensor(mesh.vertices)
    Rmat = torch.Tensor(rtk[:3,:3])
    Tmat = torch.Tensor(rtk[:3,3])
    vertices = obj_to_cam(vertices, Rmat, Tmat)

    rtk[:3,:3] = np.eye(3)
    rtk[:3,3] = 0.
    mesh = trimesh.Trimesh(vertices.numpy(), mesh.faces)
    return mesh, rtk

def main(_):

    if opts.loadname_objs == ["None"]:
        # loading from jointly finetuned model
        #loadname_objs = ["{}/{}/obj0/".format(opts.checkpoint_dir, opts.seqname), "{}/{}/obj1/".format(opts.checkpoint_dir, opts.seqname)]
        
        # count the number of directories starting with "obj%d"
        obj_dirs = sorted(glob.glob("{}/{}/obj[0-9]".format(opts.checkpoint_dir, opts.seqname)))
        loadname_objs = ["{}/{}/obj{}/".format(opts.checkpoint_dir, opts.seqname, obj_index) for obj_index in range(len(obj_dirs))]
    else:
        # loading from pretrained models or jointly trained-from-scratch model
        loadname_objs = ["{}/{}".format(opts.checkpoint_dir, loadname_obj) for loadname_obj in opts.loadname_objs]

    if opts.savename_objs == ["None"]:
        # loading jointly finetuned results (where savename_dir = seqname = config_name)
        savename_objs = ["{}/{}/obj{}/".format(opts.checkpoint_dir, opts.seqname, obj_index) for obj_index in range(len(loadname_objs))]
    else:
        # loading from pretrained models or jointly trained-from-scratch model
        savename_objs = ["{}/{}".format(opts.checkpoint_dir, savename_obj) for savename_obj in opts.savename_objs]

    opts_list = []

    for obj_index, (loadname_obj, savename_obj) in enumerate(zip(loadname_objs, savename_objs)):
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname_obj)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.model_path = "{}/params_latest.pth".format(loadname_obj)
        opts_obj.seqname = opts.seqname                              # to be used for loading the appropriate config file
        opts_obj.use_3dcomposite = opts.use_3dcomposite
        opts_obj.dist_corresp = opts.dist_corresp
        opts_obj.sample_grid3d = opts.sample_grid3d
        opts_obj.mc_threshold = opts.mc_threshold
        opts_obj.ndepth = opts.ndepth
        opts_obj.render_size = opts.render_size
        opts_obj.logname = opts.seqname                             # to be used for defining the log directory in train()

        if obj_index == len(loadname_objs) - 1:
            opts_obj.use_cc = False

        opts_list.append(opts_obj)

        # if opts.log doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy opts.log from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists(savename_obj):
            os.makedirs(savename_obj, exist_ok = True)

        if not os.path.exists("{}/{}".format(savename_obj, "opts.log")):
            shutil.copy(os.path.join(loadname_obj, "opts.log"), savename_obj)

        # if params_latest.pth doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy params_latest.pth from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("{}/{}".format(savename_obj, "params_latest.pth")):
            shutil.copy(os.path.join(loadname_obj, "params_latest.pth"), savename_obj)

        # if vars_latest.npy doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy vars_latest.npy from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("{}/{}".format(savename_obj, "vars_latest.npy")):
            shutil.copy(os.path.join(loadname_obj, "vars_latest.npy"), savename_obj)

        '''
        # if opts.log doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy opts.log from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}".format(opts.seqname, obj_index)):
            os.makedirs("logdir/{}/obj{}".format(opts.seqname, obj_index), exist_ok = True)

        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "opts.log")):
            shutil.copy(os.path.join(loadname_obj, "opts.log"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))

        # if params_latest.pth doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy params_latest.pth from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "params_latest.pth")):
            shutil.copy(os.path.join(loadname_obj, "params_latest.pth"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))

        # if vars_latest.npy doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy vars_latest.npy from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "vars_latest.npy")):
            shutil.copy(os.path.join(loadname_obj, "vars_latest.npy"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))
        '''

    # any object agnostic settings (e.g. training settings) will be taken henceforth from the background opts (index = -1)

    #trainer = v2s_trainer_objs(opts, is_eval=True)
    trainer = v2s_trainer_objs(opts_list, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model_objs(data_info)

    #dynamic_mesh = opts.flowbw or opts.lbs
    #idx_render = str_to_frame(opts.test_frames, data_info)
    #dynamic_mesh = opts_list[-1].flowbw or opts_list[-1].lbs        # won't actually be used inside eval() for multi-object version of the code base
    dynamic_mesh = True                                              # (12/30/22) dynamic_mesh is now actually used inside eval() for multi-object version of the codebase
    idx_render = str_to_frame(str(len(trainer.evalloader.dataset)), data_info)

    #trainer.model.img_size = opts.render_size
    #chunk = opts.frame_chunk
    trainer.model.img_size = opts_list[-1].render_size
    chunk = len(idx_render)

    start_time = time.time()

    for i in range(0, len(idx_render), chunk):
        #rendered_seq, aux_seq = trainer.eval(idx_render=idx_render[i:i+chunk],
        #                                     dynamic_mesh=dynamic_mesh)    
        rendered_seq, aux_seq_objs = trainer.eval(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh)    
        rendered_seq = tensor2array(rendered_seq)
        # commented out for now in order to run eval composite rendering
        #save_output(rendered_seq, aux_seq, seqname, save_flo=opts.use_corresp)
        for obj_index, (loadname_obj, savename_obj) in enumerate(zip(loadname_objs, savename_objs)):
            # saves dynamic meshes and camera poses in logdir/opts.seqname/obj%0d/
            save_output_obj(obj_index, rendered_seq, aux_seq_objs[obj_index], savename_obj, save_flo=opts.use_corresp, extract_mesh=opts.extract_mesh)
    #TODO merge the outputs

    end_time = time.time()
    print("TIME TAKEN to extract camera poses and meshes for {} objects over {} frames is {}".format(len(opts_list), len(idx_render), end_time - start_time))

if __name__ == '__main__':
    app.run(main)
