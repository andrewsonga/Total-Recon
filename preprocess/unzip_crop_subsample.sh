# usage: bash preprocess/unzip_crop_subsample.sh <src_seqname> <tgt_seqname> <start_frame> <end_frame> <src_fps> <tgt_fps>

rootdir=raw
src_seqname=$1      # name of source directory exported by record3D
                    # <src_seqname>
                    # |--- Shareable/
                    # |     |--- *.r3d

tgt_seqname=$2      # name of target directory in Total-Recon/$rootdir (will contain temporally cropped and subsampled RGBD sequence)
                    # <tgt_seqname>
                    # |--- images
                    # |--- depths
                    # |--- confs
                    # |--- metadata

start_frame=$3      # desired first frame in the raw record3D sequence
end_frame=$4        # desired end frame (inclusive) in the raw record3D sequence
src_fps=$5          # frame rate of the raw record3D sequence
tgt_fps=$6          # desired frame rate for subsampling the raw record3D sequence

# 1) unzip the .r3d file
cd $rootdir/$src_seqname/Shareable
unzip *.r3d
cd -

# 2) crop and subsample the raw record3D sequence and format it for Total-Recon
python preprocess/crop_subsample_record3d.py --src_seqname $src_seqname --tgt_seqname $tgt_seqname --start_frame $start_frame --end_frame $end_frame --src_fps $src_fps --tgt_fps $tgt_fps