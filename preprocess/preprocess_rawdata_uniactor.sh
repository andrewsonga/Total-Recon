# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

prefix=$1           # e.g. human1-mono000, cat1-mono000, dog1-mono000
ishuman=$2          # y/n
gpu=$3              # index of gpu used to run this script
isdynamic=y         # y/n
maskcamgiven=n      # y/n: whether or not object masks are given and cameras are formatted in BANMo format (OpenCV)              (4x4 RTK: [R_3x3|T_3x1]
#                               [fx,fy,px,py])

tmpdir=tmp/
camdir=cam-files/
finaloutdir=database/DAVIS/

seqname=$prefix
seqname_bkgd=$seqname-bkgd
seqname_jointft=$seqname-jointft

# 1) generate preprocessed data for the "foreground" actor
bash preprocess/preprocess_frames.sh $seqname $ishuman $isdynamic $maskcamgiven $gpu

# 2) generate preprocessed data for the "background" object
#    by copying the contents of that of the "foreground" actor
bash cp_database.sh $seqname $seqname_bkgd

# 3) generate config files for 
#    a) pretraining the "foreground" actor 
python preprocess/write_config.py \
    --seqname $seqname \
    --ishuman $ishuman \
    --metadatadir $tmpdir/$seqname/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname}/

#    b) pretraining the "background" object
python preprocess/write_config.py \
    --seqname $seqname_bkgd \
    --ishuman $ishuman \
    --metadatadir $tmpdir/$seqname/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/ \
    --rtk_path ${camdir}${seqname}/${seqname}

#    c) joint-finetuning
python preprocess/write_config.py \
    --seqname $seqname_jointft \
    --metadatadir $tmpdir/$seqname/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/ \
    --datapath_objs ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/ \
                    ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/
