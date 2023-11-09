# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================
# example run: seqname=cat2-stereo000; bash preprocess/preprocess_frames_stereo.sh $prefix n y y $gpu

prefix=$1           # e.g. human1-stereo000, cat1-stereo000, dog1-stereo000
ishuman=$2          # y/n
isdynamic=$3        # y/n
maskcamgiven=$4     # y/n: whether or not object masks are given and cameras are formatted in BANMo format (OpenCV)              (4x4 RTK: [R_3x3|T_3x1]
#                               [fx,fy,px,py])
gpu=$5              # index of gpu used to run this script

rootdir=raw/
tmpdir=tmp/
finaloutdir=database/DAVIS/
camdir=cam-files/

# create required dirs
mkdir -p $rootdir
mkdir -p $tmpdir
mkdir -p $finaloutdir
mkdir -p $camdir

# seqname (leftcam): the subdirectory inside $finaloutdir where the preprocessed data will be saved 
# seqname (rightcam): the subdirectory inside $finaloutdir where the preprocessed data will be saved
seqname_leftcam=$prefix-leftcam        # e.g. cat1-stereo000-leftcam
seqname_rightcam=$prefix-rightcam      # e.g. cat1-stereo000-rightcam
seqnames_leftright=($seqname_leftcam $seqname_rightcam)

filedir_leftcam=$rootdir/$seqname_leftcam        # points to the raw video folder (left cam)
filedir_rightcam=$rootdir/$seqname_rightcam      # points to the raw video folder (right cam)
filedirs_leftright=($filedir_leftcam $filedir_rightcam)

echo "PROCESSING $filedir_leftcam AND $filedir_rightcam"                                                    # e.g. $filedir_leftcam = raw/cat1-stereo000-leftcam
echo "PROCESSED DATA WILL BE SAVED UNDER $seqname_leftcam AND $seqname_rightcam"                            # e.g. $seqname_leftcam = database/DAVIS/cat1-stereo000-leftcam

for i in "${!seqnames_leftright[@]}"; do
    seqname=${seqnames_leftright[$i]}
    filedir=${filedirs_leftright[$i]}

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
        # copies the provided frames in raw/givenmasks/$seqname/images/ into database/DAVIS/JPEGImages/Full-Resolution/$seqname
        cp $filedir/images/* $finaloutdir/JPEGImages/Full-Resolution/$seqname/
        cp $filedir/masks/* $finaloutdir/Annotations/Full-Resolution/$seqname/
        cp $filedir/depths/* $finaloutdir/DepthMaps/Full-Resolution/$seqname/
        cp $filedir/confs/* $finaloutdir/ConfidenceMaps/Full-Resolution/$seqname/

        # copy the camera files in raw/givenmasks/$seqname/camera_rtks/ into cam-files/$prefix/
        # if $camdir/$prefix doesn't exist
        if [ ! -f $camdir/$prefix ]; then
            mkdir -p $camdir/$prefix
        fi
        cp $filedir/camera_rtks/* $camdir/$prefix/
        echo "[STEP 1] COPYING THE PROVIDED FRAMES INSIDE $filedir INTO $finaloutdir"
    else
        todir=$tmpdir/$seqname                          # tmpdir = tmp/
        rm -rf $todir                                   # todir = tmp/$seqname
        mkdir $todir
        mkdir $todir/images/
        mkdir $todir/masks/

        ###############################################################################################################
        ######################################## modified by Chonghyuk Song ###########################################
        mkdir $todir/depths/
        mkdir $todir/confs/
        mkdir $todir/metadata/
        ###############################################################################################################
        ###############################################################################################################
        
        # copies the provided frames in $rootdir/$seqname/.../images/ into $tmpdir/$seqname/images
        cp $filedir/images/* $todir/images
        ###############################################################################################################
        ######################################## modified by Chonghyuk Song ###########################################
        cp $filedir/depths/* $todir/depths
        cp $filedir/confs/* $todir/confs
        cp $filedir/metadata/* $todir/metadata
        ###############################################################################################################
        ###############################################################################################################
    fi
done
    
# computing segmentation and formatting camera parameters (e.g. changing camera poses from OpenGL to OpenCV format)
# (segmentation computation is done jointly for the leftcam and rightcam videos to sync the marking of faulty masks)
###############################################################################################################
######################################## modified by Chonghyuk Song ###########################################
if [ "$maskcamgiven" = "n" ]; then
    # sys.argv[1]: seqname_base (e.g. human1-stereo000)
    # sys.argv[2]: ishuman = 'y/n'   (whether or not this scene contains a human or pet)
    # sys.argv[3]: isdynamic = 'y/n' (whether or not this is a dynamic scene)
    CUDA_VISIBLE_DEVICES=$gpu python preprocess/mask_stereo.py $prefix $ishuman $isdynamic
    echo "[STEP 1] COMPUTED SEGMENTATIONS & FORMATTED CAMERA PARAMETERS FOR $prefix"
fi
###############################################################################################################
###############################################################################################################

# iterating over leftcam and rightcam videos
for seqname in "${seqnames_leftright[@]}"; do
    # computing densepose
    if [ "$isdynamic" = "y" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python preprocess/compute_dp.py $seqname $ishuman
        echo "[STEP 2] COMPUTED DENSEPOSE for $seqname"
    else
        echo "[STEP 2] SKIPPING DENSEPOSE"
    fi

    # computing optical flow
    cd third_party/vcnplus
    CUDA_VISIBLE_DEVICES=$gpu bash compute_flow.sh $seqname
    echo "[STEP 3] COMPUTED FLOW for $seqname"
    cd -                            # moves to previous working directory
done