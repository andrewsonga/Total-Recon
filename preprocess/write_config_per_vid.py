# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import configparser
import cv2
import glob
import pdb
import sys
import os
import json

seqname=sys.argv[1]         # e.g. human-dualrig000-leftcam
metadata_dir=sys.argv[2]     # e.g.
ishuman=sys.argv[3]         # 'y/n'
silroot='database/DAVIS/Annotations/Full-Resolution/'

config = configparser.ConfigParser()
config['data'] = {
'dframe': '1',
'init_frame': '0',
'end_frame': '-1',
'can_frame': '-1'}

total_vid = 0
img = cv2.imread('%s/%s/00000.png'%(silroot,seqname),0)
assert img is not None, "img does not exist"
num_fr = len(glob.glob('%s/%s/*.png'%(silroot,seqname)))
assert num_fr >= 16, "number of frames is less than 16"

# load cameras
# fl = max(img.shape)
with open(os.path.join(metadata_dir, 'metadata')) as f:
    data = f.read()
js = json.loads(data)       # reconstructing the data as a dictionary
K = js['K']
flx = K[0]
fly = K[4]

# define intrinsics
px = img.shape[1]//2
py = img.shape[0]//2
#camtxt = [fl,fl,px,py]
camtxt = [flx,fly,px,py]
config['data_%d'%total_vid] = {
'ishuman': ishuman,
'ks': ' '.join( [str(i) for i in camtxt] ),
'datapath': 'database/DAVIS/JPEGImages/Full-Resolution/%s/'%seqname,
}

with open('configs/%s.config'%(seqname), 'w') as configfile:
    config.write(configfile)

