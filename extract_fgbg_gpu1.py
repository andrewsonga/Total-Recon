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

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
#from nnutils.train_utils import v2s_trainer
from nnutils.train_utils_objs import v2s_trainer_objs
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 

flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
opts = flags.FLAGS
                
def save_output_obj(obj_index, rendered_seq, aux_seq, save_dir, save_flo):
    #save_dir = '%s/'%(opts.model_path.rsplit('/',1)[0])
    save_dir_obj = '%s/obj%d/'%(save_dir, obj_index)
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
    
        '''
        img_gt = rendered_seq['img'][i]
        flo_gt = rendered_seq['flo'][i]
        mask_gt = rendered_seq['sil'][i][...,0]
        flo_gt[mask_gt<=0] = 0
        img_gt[mask_gt<=0] = 1
        if save_flo: img_gt = cat_imgflo(img_gt, flo_gt)
        else: img_gt*=255
        cv2.imwrite('%s-img-gt-%05d.jpg'%(save_prefix, idx), img_gt[...,::-1])
        flo_gt_vid.append(img_gt)
        
        img_p = rendered_seq['img_coarse'][i]
        flo_p = rendered_seq['flo_coarse'][i]
        mask_gt = cv2.resize(mask_gt, flo_p.shape[:2][::-1]).astype(bool)
        flo_p[mask_gt<=0] = 0
        img_p[mask_gt<=0] = 1
        if save_flo: img_p = cat_imgflo(img_p, flo_p)
        else: img_p*=255
        cv2.imwrite('%s-img-p-%05d.jpg'%(save_prefix, idx), img_p[...,::-1])
        flo_p_vid.append(img_p)

        flo_gt = cv2.resize(flo_gt, flo_p.shape[:2])
        flo_err = np.linalg.norm( flo_p - flo_gt ,2,-1)
        flo_err_med = np.median(flo_err[mask_gt])
        flo_err[~mask_gt] = 0.
        cv2.imwrite('%s-flo-err-%05d.jpg'%(save_prefix, idx), 
                128*flo_err/flo_err_med)

        img_gt = rendered_seq['img'][i]
        img_p = rendered_seq['img_coarse'][i]
        img_gt = cv2.resize(img_gt, img_p.shape[:2][::-1])
        img_err = np.power(img_gt - img_p,2).sum(-1)
        img_err_med = np.median(img_err[mask_gt])
        img_err[~mask_gt] = 0.
        cv2.imwrite('%s-img-err-%05d.jpg'%(save_prefix, idx), 
                128*img_err/img_err_med)

#    fps = 1./(5./len(flo_p_vid))
    upsample_frame = min(30, len(flo_p_vid))    
    save_vid('%s-img-p' %(save_prefix), flo_p_vid, upsample_frame=upsample_frame)
    save_vid('%s-img-gt' %(save_prefix),flo_gt_vid,upsample_frame=upsample_frame)
    '''
    
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

    #loadname_objs = ["logdir/{}/obj0/".format(opts.seqname), "logdir/{}/obj1/".format(opts.seqname)]
    loadname_objs = ["logdir/catamelie-dualrig002-leftcam-e120-b256-ft2", "logdir/catamelie-dualrig002-fgbg-leftcam-e120-b256-ft2"]
    #loadname_objs = ["logdir/cat-pikachiu-rgbd0-ds-alignedframes-frame380-focal800-depscale0p2-depwt1-vismesh-e120-b256-ft2", "logdir/cat-pikachiu-rgbd-bkgd0-ds-alpha6to10-alignedframes-frame380-focal800-depscale0p2-vismesh-e120-b256-ft2"]
    #loadname_objs = ["logdir/andrew-dualcam000-depscale0p2-e120-b256-ft2", "logdir/andrew-dualcam-bkgd000-colmap-ds-depscale0p2-e120-b256-ft2"]
    opts_list = []

    for obj_index, loadname in enumerate(loadname_objs):
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.model_path = "{}/params_latest.pth".format(loadname)
        opts_obj.seqname = opts.seqname                              # to be used for loading the appropriate config file
        opts_obj.use_3dcomposite = opts.use_3dcomposite
        opts_obj.dist_corresp = opts.dist_corresp
        opts_obj.sample_grid3d = opts.sample_grid3d
        opts_obj.ndepth = opts.ndepth
        opts_obj.render_size = opts.render_size

        if obj_index == len(loadname_objs) - 1:
            opts_obj.use_cc = False

        opts_list.append(opts_obj)

        # if opts.log doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy opts.log from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}".format(opts.seqname, obj_index)):
            os.makedirs("logdir/{}/obj{}".format(opts.seqname, obj_index), exist_ok = True)

        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "opts.log")):
            shutil.copy(os.path.join(loadname, "opts.log"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))

        # if params_latest.pth doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy params_latest.pth from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "params_latest.pth")):
            shutil.copy(os.path.join(loadname, "params_latest.pth"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))

        # if vars_latest.npy doesn't exist inside "logdir/{}/obj0/".format(opts.seqname) (i.e. if we're loading pretrained models)
        # then copy vars_latest.npy from "loadname" into "logdir/{}/obj0/".format(opts.seqname)
        if not os.path.exists("logdir/{}/obj{}/{}".format(opts.seqname, obj_index, "vars_latest.npy")):
            shutil.copy(os.path.join(loadname, "vars_latest.npy"), "logdir/{}/obj{}/".format(opts.seqname, obj_index))

    # any object agnostic settings (e.g. training settings) will be taken henceforth from the background opts (index = -1)

    #trainer = v2s_trainer_objs(opts, is_eval=True)
    trainer = v2s_trainer_objs(opts_list, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model_objs(data_info)
    seqname=opts.seqname

    #dynamic_mesh = opts.flowbw or opts.lbs
    #idx_render = str_to_frame(opts.test_frames, data_info)
    dynamic_mesh = opts_list[-1].flowbw or opts_list[-1].lbs        # won't actually be used inside eval() for multi-object version of the code base
    idx_render = str_to_frame(str(len(trainer.evalloader.dataset)), data_info)
    #idx_render = [326]
    
#    idx_render[0] += 50
#    idx_render[0] += 374
#    idx_render[0] += 292
#    idx_render[0] += 10
#    idx_render[0] += 340
#    idx_render[0] += 440
#    idx_render[0] += 540
#    idx_render[0] += 640
#    idx_render[0] += trainer.model.data_offset[4]-4 + 37
#    idx_render[0] += 36

    #trainer.model.img_size = opts.render_size
    #chunk = opts.frame_chunk
    trainer.model.img_size = opts_list[-1].render_size
    chunk = len(idx_render)

    for i in range(0, len(idx_render), chunk):
        #rendered_seq, aux_seq = trainer.eval(idx_render=idx_render[i:i+chunk],
        #                                     dynamic_mesh=dynamic_mesh)    
        rendered_seq, aux_seq_objs = trainer.eval(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh)    
        rendered_seq = tensor2array(rendered_seq)
        # commented out for now in order to run eval composite rendering
        #save_output(rendered_seq, aux_seq, seqname, save_flo=opts.use_corresp)
        for obj_index, loadname in enumerate(loadname_objs):
            save_dir = os.path.join(opts.checkpoint_dir, opts.seqname)             # opts.checkpoint_dir = logdir/, opts.seqname = name of the config file that contains all the info
            # saves dynamic meshes and camera poses in logdir/opts.seqname/obj%0d/
            save_output_obj(obj_index, rendered_seq, aux_seq_objs[obj_index], save_dir, save_flo=opts.use_corresp)
    #TODO merge the outputs

if __name__ == '__main__':
    app.run(main)
