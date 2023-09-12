# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================
# example run: seqname=cat2-stereo000; bash preprocess/preprocess_frames_dualrig_givenmaskcam.sh $seqname n y y

rootdir=raw/
tmpdir=tmp/
prefix=$1                                       # e.g. human1-stereo000, cat1-stereo000, dog1-stereo000
filedir_leftcam=$rootdir/$prefix-leftcam        # points to the raw video folder (left cam)
filedir_rightcam=$rootdir/$prefix-rightcam      # points to the raw video folder (right cam)
filedirs_leftright=($filedir_leftcam $filedir_rightcam)
maskoutdir=$rootdir/output
finaloutdir=database/DAVIS/
camdir=cam-files/
ishuman=$2      # y/n
isdynamic=$3    # y/n
maskcamgiven=$4  # y/n
gpu=$5

# create required dirs
mkdir -p tmp
mkdir -p database/DAVIS/
mkdir -p raw/output

#counter=0
#for infile_leftcam in `ls -d $filedir_leftcam/*`; do                   # filedir = raw/$prefix (e.g. raw/andrew)

#    raw_videoname=`basename $infile_leftcam`                           # e.g. raw_videoname = video000
#    infile_rightcam=$filedir_rightcam/$raw_videoname
                # e.g. infile_rightcam = raw/human-dualrig-rightcam/video000
echo "PROCESSING $filedir_leftcam AND $filedir_rightcam"                # e.g. filedir_leftcam = raw/cat1-stereo000-leftcam

# seqname (leftcam): the folder number under which the preprocessed data will be saved under inside "database"
# seqname (rightcam): the folder number under which the preprocessed data will be saved under inside "database"
seqname_leftcam=$prefix-leftcam        # e.g. cat1-stereo000-leftcam
seqname_rightcam=$prefix-rightcam      # e.g. cat1-stereo000-rightcam
seqnames_leftright=($seqname_leftcam $seqname_rightcam)

echo "PROCESSED DATA WILL BE SAVED UNDER $seqname_leftcam AND $seqname_rightcam"

for i in "${!seqnames_leftright[@]}"; do
    seqname=${seqnames_leftright[$i]}
    filedir=${filedirs_leftright[$i]}
    #infile=${infiles_leftright[$i]}

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
        # if camdir doesn't exist
        if [ ! -f $camdir/$prefix ]; then
            mkdir -p $camdir/$prefix
        fi
        cp $filedir/camera_rtks/* $camdir/$prefix/
        echo "[STEP 1] copying the provided frames inside $filedir into $finaloutdir"
    else
        # copies the provided frames in /raw/$seqname/.../images/ into tmp/$seqname_num/images
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
        
        # copies the provided frames in /raw/$seqname/.../images/ into tmp/$seqname_num/images
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
    
# segmentation (jointly for the leftcam and rightcam videos to sync the discarding of faulty masks)
###############################################################################################################
######################################## modified by Chonghyuk Song ###########################################
# $prefix$(printf "%03d" $counter): e.g. human-dualrig000
if [ "$maskcamgiven" = "n" ]; then
    # sys.argv[1]: seqname_base (e.g. human-stereo000)
    # sys.argv[2]: ishuman = 'y/n'
    # sys.argv[3]: isdynamic = 'y/n' (whether or not this is a dynamic scene)
    CUDA_VISIBLE_DEVICES=$gpu python preprocess/mask_dualrig.py $prefix $ishuman $isdynamic
    echo "[STEP 1] COMPUTED SEGMENTATIONS for $prefix"
fi
###############################################################################################################
###############################################################################################################

# iterating over leftcam and rightcam videos
for seqname in "${seqnames_leftright[@]}"; do
    # densepose
    if [ "$isdynamic" = "y" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python preprocess/compute_dp.py $seqname $ishuman
        echo "[STEP 2] COMPUTED DENSEPOSE for $seqname"
    else
        echo SKIPPING DENSEPOSE
    fi

    # flow
    cd third_party/vcnplus
    CUDA_VISIBLE_DEVICES=$gpu bash compute_flow.sh $seqname
    echo "[STEP 3] COMPUTED FLOW for $seqname"
    cd -                            # moves to previous working directory
done