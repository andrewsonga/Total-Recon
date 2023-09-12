# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# prefix=humandog-stereo000 or humancat-stereo000
prefix=$1
gpu=$2
isdynamic=y        # y/n
maskcamgiven=y
rootdir=raw
tmpdir=tmp

# 1) make tmp if it doesn't exist
mkdir -p $tmpdir

# 3) generate preprocessed data for the "human" actor
#    this version of the dataset will include masks for the human actor, and densepose features extracted for the human  actor
bash preprocess/preprocess_frames_dualrig_givenmaskcam.sh $prefix-human y $isdynamic $maskcamgiven $gpu 

# 4) generate preprocessed data for the "animal" actor
#    this version of the dataset will include masks for the animal actor, and densepose features extracted for the animal (quadruped) actor
bash preprocess/preprocess_frames_dualrig_givenmaskcam.sh $prefix-animal n $isdynamic $maskcamgiven $gpu

# 5) generate preprocessed data for the "background" object
#    by copying the contents of that of the "human" actor
bash cp_database.sh $prefix-human-leftcam $prefix-bkgd-leftcam
bash cp_database.sh $prefix-human-rightcam $prefix-bkgd-rightcam

# 6) generate "uncropped", "full-res" version of preprocessed data for each actor
bash cp_database.sh $prefix-human-leftcam $prefix-human-uncropped-leftcam
bash cp_database.sh $prefix-human-rightcam $prefix-human-uncropped-rightcam
bash cp_database.sh $prefix-animal-leftcam $prefix-animal-uncropped-leftcam
bash cp_database.sh $prefix-animal-rightcam $prefix-animal-uncropped-rightcam