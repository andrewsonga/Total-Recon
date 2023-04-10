from absl import app
from absl import flags
import cv2
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from nnutils.train_utils_objs import v2s_trainer_objs
opts = flags.FLAGS

#############################################################
################ modified by Chonghyuk Song #################    
flags.DEFINE_bool('recon_bkgd', False, 'whether or not to reconstruct just the background')
flags.DEFINE_string('optslog', '', 'path to optslog')
#############################################################
#############################################################

def main(_):
    if opts.local_rank==0:
        log_file = osp.join(opts.checkpoint_dir, opts.optslog, "opts.log")
        if not osp.exists(osp.join(opts.checkpoint_dir, opts.optslog)): 
            os.makedirs(osp.join(opts.checkpoint_dir, opts.optslog))
        opts.append_flags_into_file(log_file)

if __name__ == '__main__':
    app.run(main)
