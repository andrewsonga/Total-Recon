# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Chonghyuk Song’s modifications are Copyright (c) 2023, Chonghyuk Song. All rights reserved.
# Chonghyuk Song’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================
# example run: seqname=cat1-mono000; bash preprocess/preprocess_frames.sh $seqname n y $gpu

seqname=$1          # the subdirectory inside $finaloutdir where the preprocessed data will be saved (e.g. human1-mono000, cat1-mono000, dog1-mono000)
ishuman=$2          # y/n
isdynamic=$3        # y/n
maskcamgiven=$4     # y/n: whether or not object masks are given and cameras are formatted in BANMo format (OpenCV)              (4x4 RTK: [R_3x3|T_3x1]
#                               [fx,fy,px,py])
gpu=$5              # index of gpu used to run this script

echo GPU IS $gpu

rootdir=raw/
tmpdir=tmp/
finaloutdir=database/DAVIS/
camdir=cam-files/

# 1) create required dirs
mkdir -p $rootdir
mkdir -p $tmpdir
mkdir -p $finaloutdir
mkdir -p $camdir

filedir=$rootdir/$seqname                                                   # points to the raw video folder
echo "PROCESSING $filedir"                                                  # e.g. $filedir = raw/cat1-mono000
echo "PROCESSED DATA WILL BE SAVED UNDER $seqname"                          # e.g. $seqname = database/DAVIS/cat1-mono000

# 2) clear existing directories inside $finaloutdir with the same name
rm -rf $finaloutdir/JPEGImages/Full-Resolution/$seqname  
rm -rf $finaloutdir/Annotations/Full-Resolution/$seqname 
rm -rf $finaloutdir/Densepose/Full-Resolution/$seqname
rm -rf $finaloutdir/DepthMaps/Full-Resolution/$seqname
rm -rf $finaloutdir/ConfidenceMaps/Full-Resolution/$seqname   
mkdir -p $finaloutdir/JPEGImages/Full-Resolution/$seqname
mkdir -p $finaloutdir/Annotations/Full-Resolution/$seqname
mkdir -p $finaloutdir/Densepose/Full-Resolution/$seqname
mkdir -p $finaloutdir/DepthMaps/Full-Resolution/$seqname
mkdir -p $finaloutdir/ConfidenceMaps/Full-Resolution/$seqname

if [ "$maskcamgiven" = "y" ]; then
    # assumes the given masks are placed inside $filedir/masks
    # and that the given cameras (in the correct format) are placed inside $filedir/camera_rtks

    # 3) copies the provided frames in raw/givenmasks/$seqname/images/ into database/DAVIS/JPEGImages/Full-Resolution/$seqname
    cp $filedir/images/* $finaloutdir/JPEGImages/Full-Resolution/$seqname/
    cp $filedir/masks/* $finaloutdir/Annotations/Full-Resolution/$seqname/
    cp $filedir/depths/* $finaloutdir/DepthMaps/Full-Resolution/$seqname/
    cp $filedir/confs/* $finaloutdir/ConfidenceMaps/Full-Resolution/$seqname/

    # 4) copy the camera files in $filedir/camera_rtks into $camdir/$seqname/
    # if $camdir/$seqname doesn't exist
    if [ ! -f $camdir/$seqname ]; then
        mkdir -p $camdir/$seqname
    fi
    cp $filedir/camera_rtks/* $camdir/$seqname/
    echo "[STEP 1] COPYING THE PROVIDED FRAMES INSIDE $filedir INTO $finaloutdir"
else
    # 3) clear existing directories inside $tmpdir with the same name
    todir=$tmpdir/$seqname                          # tmpdir = tmp/
    rm -rf $todir                                   # todir = tmp/$seqname
    mkdir $todir
    mkdir $todir/images/
    mkdir $todir/masks/
    mkdir $todir/depths/
    mkdir $todir/confs/
    mkdir $todir/metadata/

    # 4) copies the provided frames in $rootdir/$seqname/.../images/ into $tmpdir/$seqname/images
    cp $filedir/images/* $todir/images
    cp $filedir/depths/* $todir/depths
    cp $filedir/confs/* $todir/confs
    cp $filedir/metadata/* $todir/metadata
fi
    
# 5) computing segmentation and formatting camera parameters (e.g. changing camera poses from OpenGL to OpenCV format)
# sys.argv[1]: seqname (e.g. human1-mono000)
# sys.argv[2]: ishuman = 'y/n'   (whether or not this scene contains a human or pet)
# sys.argv[3]: isdynamic = 'y/n' (whether or not this is a dynamic scene)
if [ "$maskcamgiven" = "n" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python preprocess/mask.py $seqname $ishuman $isdynamic
    echo "[STEP 1] COMPUTED SEGMENTATIONS & FORMATTED CAMERA PARAMETERS FOR $seqname"
fi

# 6) computing densepose
if [ "$isdynamic" = "y" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python preprocess/compute_dp.py $seqname $ishuman
    echo "[STEP 2] COMPUTED DENSEPOSE for $seqname"
else
    echo "[STEP 2] SKIPPING DENSEPOSE"
fi

# 7) computing optical flow
cd third_party/vcnplus
CUDA_VISIBLE_DEVICES=$gpu bash compute_flow.sh $seqname
echo "[STEP 3] COMPUTED FLOW for $seqname"
cd -                            # moves to previous working directory

# 8) generate config file
