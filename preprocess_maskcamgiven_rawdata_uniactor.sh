# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

# prefix=human1-stereo000 or dog1-stereo000
prefix=$1
ishuman=$2         # y/n
gpu=$3
isdynamic=y        # y/n
maskcamgiven=y
rootdir=raw
tmpdir=tmp

# 1) make tmp if it doesn't exist
mkdir -p $tmpdir

# 2) generate preprocessed data for the "foreground" actor
bash preprocess/preprocess_frames_dualrig_givenmaskcam.sh $prefix $ishuman $isdynamic $maskcamgiven $gpu

# 3) generate preprocessed data for the "background" object
#    by copying the contents of that of the "foreground" actor
bash cp_database.sh $prefix-leftcam $prefix-bkgd-leftcam
bash cp_database.sh $prefix-rightcam $prefix-bkgd-rightcam