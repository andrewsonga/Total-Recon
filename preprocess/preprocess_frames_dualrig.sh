# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

rootdir=raw/
tmpdir=tmp/
prefix=$1                                       # e.g. human-dualrig, cat-dualrig, dog-dualrig
filedir_leftcam=$rootdir/$prefix-leftcam        # points to the raw video folder (left cam)
filedir_rightcam=$rootdir/$prefix-rightcam      # points to the raw video folder (right cam)
filedirs_leftright=($filedir_leftcam $filedir_rightcam)
maskoutdir=$rootdir/output
finaloutdir=database/DAVIS/
ishuman=$2 # y/n
isdynamic=$3 # y/n

# create required dirs
mkdir -p tmp
mkdir -p database/DAVIS/
mkdir -p raw/output

counter=0
for infile_leftcam in `ls -d $filedir_leftcam/*`; do                # filedir = raw/$prefix (e.g. raw/andrew)

    raw_videoname=`basename $infile_leftcam`                        # e.g. raw_videoname = video000
    infile_rightcam=$filedir_rightcam/$raw_videoname                # e.g. infile_rightcam = raw/human-dualrig-rightcam/video000
    echo "PROCESSING $infile_leftcam AND $infile_leftcam"           # e.g. infile_leftcam = raw/human-dualrig-leftcam/video000
  
    # seqname (leftcam): the folder number under which the preprocessed data will be saved under inside "database"
    # seqname (rightcam): the folder number under which the preprocessed data will be saved under inside "database"
    seqname_leftcam=$prefix$(printf "%03d" $counter)-leftcam        # e.g. human-dualrig000-leftcam
    seqname_rightcam=$prefix$(printf "%03d" $counter)-rightcam      # e.g. human-dualrig000-rightcam
  
    echo "PROCESSED DATA WILL BE SAVED UNDER $seqname_leftcam AND $seqname_rightcam"

    # iterating over leftcam and rightcam videos
    infiles_leftright=($infile_leftcam $infile_rightcam)
    seqnames_leftright=($seqname_leftcam $seqname_rightcam)

    #for seqname in "${seqnames_leftright[@]}"
    for i in "${!seqnames_leftright[@]}"; do
        seqname=${seqnames_leftright[$i]}
        infile=${infiles_leftright[$i]}

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
        cp $infile/images/* $todir/images
        ###############################################################################################################
        ######################################## modified by Chonghyuk Song ###########################################
        cp $infile/depths/* $todir/depths
        cp $infile/confs/* $todir/confs
        cp $infile/metadata/* $todir/metadata
        ###############################################################################################################
        ###############################################################################################################

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
    done
    
    # segmentation (jointly for the leftcam and rightcam videos to sync the discarding of faulty masks)
    ###############################################################################################################
    ######################################## modified by Chonghyuk Song ###########################################
    # $prefix$(printf "%03d" $counter): e.g. human-dualrig000
    
    python preprocess/mask_dualrig.py $prefix$(printf "%03d" $counter) $ishuman $prefix $isdynamic
    echo "[STEP 1] COMPUTED SEGMENTATIONS for $prefix$(printf "%03d" $counter)"
    ###############################################################################################################
    ###############################################################################################################

    # iterating over leftcam and rightcam videos
    for seqname in "${seqnames_leftright[@]}"; do
        # densepose
        if [ "$isdynamic" = "y" ]; then
            python preprocess/compute_dp.py $seqname $ishuman
            echo "[STEP 2] COMPUTED DENSEPOSE for $seqname"
        else
            echo SKIPPING DENSEPOSE
        fi

        # flow
        cd third_party/vcnplus
        bash compute_flow.sh $seqname
        echo "[STEP 3] COMPUTED FLOW for $seqname"
        cd -                            # moves to previous working directory

        # Optionally run SfM for initial root pose
        # bash preprocess/colmap_to_data.sh $seqname $ishuman
    done
    counter=$((counter+1))
done

# write config file (one that includes all videos with the same prefix) and a config file for each video
#python preprocess/write_config.py ${seqname::-3} $ishuman
python preprocess/write_config.py $prefix $ishuman

counter=0
for infile_leftcam in `ls -d $filedir_leftcam/*`; do              # filedir = raw/$prefix (e.g. raw/andrew)
    
    # where as "write_config.py" takes all videos with the same prefix (e.g. andrew-dualcam) and compiles them into a single .config (as per the problem setting of banmo of using multiple separate videos)
    # "write_config_dualrig.py" consider a single pair of corresponding videos (left-rightcam) and makes a .config for each of pair of videos
    raw_videoname=`basename $infile_leftcam`                        # e.g. raw_videoname = video000
    infile_rightcam=$filedir_rightcam/$raw_videoname                 # e.g. infile_rightcam = raw/human-dualrig-rightcam/video000

    metadatadir_leftcam=$infile_leftcam/metadata
    metadatadir_rightcam=$infile_rightcam/metadata

    python preprocess/write_config_per_vid.py $prefix$(printf "%03d" $counter)-leftcam $metadatadir_leftcam $ishuman
    python preprocess/write_config_per_vid.py $prefix$(printf "%03d" $counter)-rightcam $metadatadir_rightcam $ishuman
    echo "[STEP 4] GENERATED CONFIG FILE for $prefix$(printf "%03d" $counter)"

    counter=$((counter+1))
done