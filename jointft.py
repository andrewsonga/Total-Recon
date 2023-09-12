# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import app
from absl import flags
import cv2
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import copy
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import glob

#########################################################################################################
#################################### modified by Chonghyuk Song #########################################
from nnutils.train_utils import v2s_trainer_objs
#########################################################################################################
#########################################################################################################

#flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
flags.DEFINE_multi_string('loadname_objs', 'None', 'name of folder inside \logdir to load into fg banmo object')
flags.DEFINE_bool('freeze_shape_bkgd',True,'whether or not to freeze the background shape during joint finetuning')
flags.DEFINE_bool('freeze_cam_bkgd',False,'whether or not freeze the backgrounds camera parameters (root-body pose and intrinsics) during joint finetuning')
opts = flags.FLAGS

def main(_): 

    #########################################################################################################
    #################################### modified by Chonghyuk Song #########################################
    if opts.loadname_objs == ["None"]:
        # loading from jointly finetuned model
        # count the number of directories starting with "obj%d"
        obj_dirs = glob.glob("{}/{}/obj[0-9]".format(opts.checkpoint_dir, opts.seqname))
        loadname_objs = ["{}/{}/obj{}/".format(opts.checkpoint_dir, opts.seqname, obj_index) for obj_index in range(len(obj_dirs))]
    else:
        # loading from pretrained models or jointly trained-from-scratch model
        loadname_objs = ["{}/{}".format(opts.checkpoint_dir, loadname_obj) for loadname_obj in opts.loadname_objs]

    opts_list = []

    for obj_index, loadname in enumerate(loadname_objs):
        
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(loadname)])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.local_rank = opts.local_rank           # to account for local_rank automatically set by def main() function
        opts_obj.ngpu = opts.ngpu                       # to account for ngpu automatically set by def main() function
        opts_obj.pose_cnn_path = ''                     # to prevent downstream scripts from running under the assumption we're using posenet (we're loading from pretrained cameras)        
        
        opts_obj.model_path = "{}/params_latest.pth".format(loadname)
        opts_obj.eikonal_loss = opts.eikonal_loss
        opts_obj.eikonal_loss2 = opts.eikonal_loss2
        opts_obj.dense_trunc_eikonal_loss = opts.dense_trunc_eikonal_loss
        opts_obj.dist_corresp = opts.dist_corresp
        
        if obj_index < len(loadname_objs) - 1:          # obj_index == len(loadname_objs) - 1 -> bkgd
            opts_obj.use_unc = opts.use_unc
        else:
            opts_obj.use_unc = False                    # let's not use active sampling on bkgd for now and focus on active sampling for the fg objects
        opts_obj.use_ent = opts.use_ent
        opts_obj.use_proj = opts.use_proj
        opts_obj.use_embed = opts.use_embed

        opts_obj.nf_reset = opts.nf_reset
        opts_obj.bound_reset = opts.bound_reset

        opts_obj.ent_wt = opts.ent_wt
        opts_obj.dep_wt = opts.dep_wt
        opts_obj.sil_wt = opts.sil_wt
        opts_obj.flow_wt = opts.flow_wt
        if obj_index < len(loadname_objs) - 1:          # obj_index == len(loadname_objs) - 1 -> bkgd
            opts_obj.feat_wt = opts.feat_wt
            opts_obj.frnd_wt = opts.frnd_wt
            opts_obj.proj_wt = opts.proj_wt       
            
        opts_obj.num_epochs = opts.num_epochs
        opts_obj.reset_beta = opts.reset_beta
        opts_obj.rm_novp = opts.rm_novp                # (09/11 removing this flag for long joint fine-tuning)
        opts_obj.use_3dcomposite = opts.use_3dcomposite

        # allow for nofreeze_root, nofreeze_shape for only the foreground objects
        if obj_index < len(loadname_objs) - 1:          # obj_index == len(loadname_objs) - 1 -> bkgd
            opts_obj.freeze_shape = opts.freeze_shape
            opts_obj.freeze_root = opts.freeze_root     # added for joint finetuning, s.t. fg and bkgd frames can be tuned to be consistently placed w.r.t. each other
            opts_obj.ks_opt = opts.ks_opt
        else:
            if opts.freeze_shape_bkgd:
                opts_obj.freeze_shape = True                    # for our complete model (on most sequences)
            else:
                opts_obj.freeze_shape = False                   # for our complete model on the cat1-stereo000 sequence

            if opts.freeze_cam_bkgd:
                opts_obj.freeze_root = True                     # for ablated method (freezing camera pose)
                opts_obj.ks_opt = False                         # for ablated method (freezing intrinsics)
            else:
                opts_obj.freeze_root = False                    # for our complete model
                opts_obj.ks_opt = True                          # for our complete model
            
        opts_obj.freeze_coarse = opts.freeze_coarse
        opts_obj.freeze_body_mlp = opts.freeze_body_mlp
        opts_obj.freeze_coarse_minus_beta = opts.freeze_coarse_minus_beta
        opts_obj.freeze_cvf = opts.freeze_cvf
        opts_obj.samertktraj_opt = opts.samertktraj_opt

        opts_obj.learning_rate = opts.learning_rate
        opts_obj.batch_size = opts.batch_size

        opts_obj.seqname = opts.seqname                             # to be used for loading the appropriate config file
        opts_obj.logname = opts.seqname                             # to be used for defining the log directory in train()

        opts_list.append(opts_obj)

    torch.cuda.set_device(opts_list[-1].local_rank)
    world_size = opts_list[-1].ngpu
    torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=opts_list[-1].local_rank
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    
    #########################################################################################################
    #################################### modified by Chonghyuk Song #########################################
    trainer = v2s_trainer_objs(opts_list)
    data_info = trainer.init_dataset()    
    trainer.define_model_objs(data_info)
    #########################################################################################################
    #########################################################################################################
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
