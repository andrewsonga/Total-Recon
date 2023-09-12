import numpy as np
import glob
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate identity poses (for when we ablate PoseNet)")
    parser.add_argument('--src_rtk_path', default='cam-files/', help='(e.g. cam-files/catpika-dualrig-fgbg001/catpika-dualrig-fgbg001-leftcam) directory from which we source 1) number of camera frames, 2) intrinsic parameters')
    parser.add_argument('--refcam', default='leftcam', help='(= `leftcam` or `rightcam`) the reference camera whose RGBD images are used to train our fg-bgkd model')

    args = parser.parse_args()

# 1) read the intrinsics and compute the number of frames for the sequence in question
src_rtk_path_list = sorted(glob.glob(args.src_rtk_path+'*.txt'))

# extract number of frames and camera intrinsics
num_frames = len(src_rtk_path_list)
rtk_0 = np.loadtxt(src_rtk_path_list[0])
ks = rtk_0[-1]

# define identity pose
identity_pose = np.eye(4)
identity_pose[-1] = ks          # set final row to be the intrinsics
identity_pose[2, 3] = 3.        # set tmat = [0, 0, 3.].T

# 2) write into the target directory the identity camera poses (one for each frame)
for src_rtk_path in src_rtk_path_list:

    assert(src_rtk_path.count(args.refcam) == 1)
    tgt_rtk_path = src_rtk_path.replace(args.refcam, "identitypose-"+args.refcam)
    
    print("saving identity pose to {}".format(tgt_rtk_path))

    with open(tgt_rtk_path, 'w') as f:
        for row in identity_pose:
            f.write(" ".join([str(x) for x in row]) + "\n")