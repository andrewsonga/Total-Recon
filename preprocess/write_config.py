# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

import configparser
import argparse
import cv2
import glob
import pdb
import os
import json

parser = argparse.ArgumentParser(description="generate config file")
parser.add_argument('--seqname', default='None', help='name of the config file')
parser.add_argument('--ishuman', default='None', help='whether or not the scene contains of human actor: y (yes) or n (no)')
parser.add_argument('--metadatadir', default='None', help='directory of metadata file that contains the intrinsics')
parser.add_argument('--datapath', default='None', help='datapath for pretraining object fields')
parser.add_argument('--datapath_objs', default=[], nargs="+", help='datapaths for joint-finetuning in order of object index')
parser.add_argument('--rtk_path', default='None', help='')
args = parser.parse_args()

config = configparser.ConfigParser()
config['data'] = {
'dframe': '1',
'init_frame': '0',
'end_frame': '-1',
'can_frame': '-1'}

total_vid = 0
img = cv2.imread('%s/00000.jpg'%(args.datapath))
assert img is not None, "img does not exist"
num_fr = len(glob.glob('%s/*.jpg'%(args.datapath)))
assert num_fr >= 16, "number of frames is less than 16"

data_dict = {}

if not args.ishuman == "None":
    data_dict["ishuman"] = args.ishuman
if not args.metadatadir == "None":
    # load camera intrinsic parameters
    with open(os.path.join(args.metadatadir, 'metadata')) as f:
        data = f.read()
    js = json.loads(data)       # reconstructing the data as a dictionary
    K = js['K']

    flx = K[0]
    px = K[6]                   # x-dir principal point offset taken from args.metadatadir
    #px = img.shape[1]//2       # x-dir principal point offset computed as 1/2 of img width (used for the RGBD sequences provided by Total-Recon)
    fly = K[4]
    py = K[7]                   # y-dir principal point offset taken from args.metadatadir
    #py = img.shape[0]//2       # y-dir principal point offset computed as 1/2 of img height (used for the RGBD sequences provided by Total-Recon)
    intrinsics = [flx,fly,px,py]
    data_dict["ks"] = ' '.join([str(i) for i in intrinsics])
if not args.datapath == "None":
    data_dict["datapath"] = args.datapath
for obj_index, datapath_obj in enumerate(args.datapath_objs):
    data_dict["datapath{}".format(obj_index)] = datapath_obj
if not args.rtk_path == "None":
    data_dict["rtk_path"] = args.rtk_path

config['data_%d'%total_vid] = data_dict

with open('configs/%s.config'%(args.seqname), 'w') as configfile:
    config.write(configfile)
