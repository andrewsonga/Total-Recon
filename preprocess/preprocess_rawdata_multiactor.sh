# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

prefix=$1          # e.g. humandog-mono000
gpu=$2             # index of gpu used to run this script    
isdynamic=y        # y/n
maskcamgiven=n     # y/n: whether or not object masks are given and cameras are formatted in BANMo format (OpenCV)              (4x4 RTK: [R_3x3|T_3x1]
#                               [fx,fy,px,py])

rootdir=raw/
tmpdir=tmp/
camdir=cam-files/
finaloutdir=database/DAVIS/

seqname=$prefix
seqname_human=$seqname-human
seqname_human_uncropped=$seqname_human-uncropped
seqname_animal=$seqname-animal
seqname_animal_uncropped=$seqname_animal-uncropped
seqname_bkgd=$seqname-bkgd
seqname_jointft=$seqname-jointft

# 1) make copies of raw dataset for per-agent data preprocessing
echo making copies of raw dataset for per-agent data preprocessing
rm -rf $rootdir/$seqname_human
rm -rf $rootdir/$seqname_animal
cp -r $rootdir/$seqname $rootdir/$seqname_human
cp -r $rootdir/$seqname $rootdir/$seqname_animal

# 2) generate preprocessed data for the "human" actor
#    this version of the dataset will include masks and densepose features extracted for the human actor
bash preprocess/preprocess_frames.sh $seqname_human y $isdynamic $maskcamgiven $gpu

# 3) generate preprocessed data for the "animal" actor
#    this version of the dataset will include masks and densepose features extracted for the animal (quadruped) actor
bash preprocess/preprocess_frames.sh $seqname_animal n $isdynamic $maskcamgiven $gpu

# 4) generate preprocessed data for the "background" object
#    by copying the contents of that of the "human" actor
bash cp_database.sh $seqname_human $seqname_bkgd

# NOTE: this part is only required for formatting preprocessed data, which is used for training
# 5) generate "uncropped, full-res" version of preprocessed data for each actor
bash cp_database.sh $seqname_human $seqname_human_uncropped
bash cp_database.sh $seqname_animal $seqname_animal_uncropped

# 6) generate config files for:
#    a) pretraining the "human" actor
python preprocess/write_config.py \
    --seqname $seqname_human \
    --ishuman y \
    --metadatadir $tmpdir/$seqname_human/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_human}/

#    b) pretraining the "animal" actor
python preprocess/write_config.py \
    --seqname $seqname_animal \
    --ishuman n \
    --metadatadir $tmpdir/$seqname_animal/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_animal}/

#    c) pretraining the "background" object (setting --ishuman to "y" is important to prevent runtime errors!)
python preprocess/write_config.py \
    --seqname $seqname_bkgd \
    --ishuman y \
    --metadatadir $tmpdir/$seqname_human/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/ \
    --rtk_path ${camdir}${seqname_human}/${seqname_human}

#    d) joint-finetuning
python preprocess/write_config.py \
    --seqname $seqname_jointft \
    --metadatadir $tmpdir/$seqname_human/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_human}/ \
    --datapath_objs ${finaloutdir}JPEGImages/Full-Resolution/${seqname_human_uncropped}/ \
                    ${finaloutdir}JPEGImages/Full-Resolution/${seqname_animal_uncropped}/ \
                    ${finaloutdir}JPEGImages/Full-Resolution/${seqname_bkgd}/

#    e) for each actor, formatting preprocessed data for an "uncropped, full-res" version
#       (the script for formatting preprocessed data requires a config file)
python preprocess/write_config.py \
    --seqname $seqname_human_uncropped \
    --ishuman y \
    --metadatadir $tmpdir/$seqname_human/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_human_uncropped}/

python preprocess/write_config.py \
    --seqname $seqname_animal_uncropped \
    --ishuman n \
    --metadatadir $tmpdir/$seqname_animal/metadata/ \
    --datapath ${finaloutdir}JPEGImages/Full-Resolution/${seqname_animal_uncropped}/
