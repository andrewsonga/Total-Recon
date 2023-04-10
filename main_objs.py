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

#########################################################################################################
#################################### modified by Chonghyuk Song #########################################
#from nnutils.train_utils import v2s_trainer
from nnutils.train_utils_objs import v2s_trainer_objs
#########################################################################################################
#########################################################################################################

flags.DEFINE_bool('recon_bkgd',False,'whether or not object in question is reconstructing the background (determines self.crop_factor in BaseDataset')
flags.DEFINE_multi_string('loadname_objs', None, 'names of folder inside \logdir to load into each banmo object separate by spaces')
opts = flags.FLAGS

def main(_): 
    #########################################################################################################
    #################################### modified by Chonghyuk Song #########################################
    opts_list = []
    #print("OPTS: {}".format(opts.flag_values_dict()))
    #print("OPTS use_embed: {}".format(opts.get_flag_value('use_embed', True)))

    for loadname in opts.loadname_objs:
        
        opts_obj = copy.deepcopy(opts)
        args_obj = opts_obj.read_flags_from_files(['--flagfile={}/opts.log'.format(osp.join(opts.checkpoint_dir, loadname))])
        opts_obj._parse_args(args_obj, known_only=True)
        opts_obj.local_rank = opts.local_rank           # to account for local_rank automatically set by def main() function
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
    
    trainer = v2s_trainer_objs(opts_list)
    data_info = trainer.init_dataset()    
    trainer.define_model_objs(data_info)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
