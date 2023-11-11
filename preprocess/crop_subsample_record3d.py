import os
import glob
import shutil
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='crop and subsample raw record3d sequence and format it for Total-Recon')
parser.add_argument('--rootdir', default='raw', type=str,
                    help='subdirectory inside Total-Recon/ where the raw data is stored')
parser.add_argument('--src_seqname', default='seqname_src', type=str,
                    help='name of source directory exported by record3D (contains the subdirectory Shareable/)')
parser.add_argument('--tgt_seqname', default='seqname_tgt', type=str,
                    help='name of target directory in Total-Recon/$rootdir (will contain temporally cropped and subsampled RGBD sequence)')
parser.add_argument('--start_frame', default=0, type=int,
                    help='desired first frame in the raw record3D sequence')
parser.add_argument('--end_frame', default=-1, type=int,
                    help='desired end frame (inclusive) in the raw record3D sequence')
parser.add_argument('--src_fps', default=30, type=int,
                    help='frame rate of the raw record3D sequence')
parser.add_argument('--tgt_fps', default=10, type=int,
                    help='desired frame rate for subsampliong the raw record3d sequence')

def main():
    args = parser.parse_args()
    rootdir = args.rootdir
    src_seqname = args.src_seqname
    tgt_seqname = args.tgt_seqname
    start_frame = args.start_frame
    end_frame = args.end_frame
    src_fps = args.src_fps
    tgt_fps = args.tgt_fps

    # 1) create target directory
    if not os.path.exists(os.path.join(rootdir, tgt_seqname)):
        os.makedirs(os.path.join(rootdir, tgt_seqname))
        os.makedirs(os.path.join(rootdir, tgt_seqname, "images"))
        os.makedirs(os.path.join(rootdir, tgt_seqname, "depths"))
        os.makedirs(os.path.join(rootdir, tgt_seqname, "confs"))
        os.makedirs(os.path.join(rootdir, tgt_seqname, "metadata"))

    # 2) read the metadata
    metadata_src_dir = os.path.join(rootdir, src_seqname, "Shareable", "metadata")
    with open(metadata_src_dir) as f:
        metadata = f.read()

    metadata = json.loads(metadata)                                             # reconstructing metadata as a dictionary
    cam2world_src = np.array(metadata['poses'])

    # 3) compute subsampled indices based on start_frame, end_frame, src_fps, and tgt_fps
    if end_frame == -1:
        end_frame = cam2world_src.shape[0]-1

    assert(src_fps % tgt_fps == 0)                                              # assert src_fps is an integer multiple of tgt_fps
    num_skipframes = int(float(src_fps) / float(tgt_fps))

    frameids_subsampled = np.arange(start_frame, end_frame+1, num_skipframes)   # inclusive end_frame

    assert(np.all(np.diff(frameids_subsampled) > 0))                            # assert that frameids_subsampled is already sorted in increasing order

    # 4) modify metadata and write it to the target directory
    metadata['fps'] = tgt_fps     
    metadata['poses'] = cam2world_src[frameids_subsampled, ...].tolist()

    with open(os.path.join(rootdir, tgt_seqname, "metadata", "metadata"), "w") as f:
        json.dump(metadata, f)

    # 5) copy the subsampled images, depths, and confs to the target directory
    for frameid_tgt, frameid_subsampled in enumerate(frameids_subsampled):
        # cp images
        shutil.copy2(os.path.join(rootdir, src_seqname, "Shareable", "rgbd", "%d.jpg"%(frameid_subsampled)), os.path.join(rootdir, tgt_seqname, "images", "%05d.jpg"%(frameid_tgt)))
                     
        # cp depths
        shutil.copy2(os.path.join(rootdir, src_seqname, "Shareable", "rgbd", "%d.depth"%(frameid_subsampled)), os.path.join(rootdir, tgt_seqname, "depths", "%05d.depth"%(frameid_tgt)))

        # cp confs
        shutil.copy2(os.path.join(rootdir, src_seqname, "Shareable", "rgbd", "%d.conf"%(frameid_subsampled)), os.path.join(rootdir, tgt_seqname, "confs", "%05d.conf"%(frameid_tgt)))
    
    # 6) compute the number of files ending with the appropriate extension e.g. .jpg for images, .depth for depths, and .conf for confs) inside each target subdirectory: they should also be the same
    num_images = len(glob.glob(os.path.join(rootdir, tgt_seqname, "images", "*.jpg")))
    num_depths = len(glob.glob(os.path.join(rootdir, tgt_seqname, "depths", "*.depth")))
    num_confs = len(glob.glob(os.path.join(rootdir, tgt_seqname, "confs", "*.conf")))
    num_poses = len(metadata["poses"])

    assert num_images == num_depths == num_confs == num_poses

if __name__ == "__main__":
    main()