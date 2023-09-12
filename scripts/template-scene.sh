# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

gpus=$1
seqname=$2
addr=$3
dep_wt=$4
sil_wt=$5
flow_wt=$6
ent_wt=$7
add_args=${*: 8:$#-1}
num_epochs=30
batch_size=256

# savename is the name of folder where models are saved
# seqname is the name of the folder where the jointly-finetuned model is saved, as well as the name of config file from which we get the datapath and the rtkpaths
# we no longer use the projection loss, feature reprojection loss, and the feature reconstruction loss during joint-finetuning
bash scripts/template-scene-mgpu.sh $gpus \
    $seqname $addr --num_epochs $num_epochs \
  --lineload --batch_size $batch_size \
  --nf_reset 0 --bound_reset 0  \
  --nofreeze_root --nofreeze_shape \
  --use_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --proj_wt 0 --feat_wt 0 --frnd_wt 0 --dist_corresp --norm_novp --use_ent --ent_wt $ent_wt \
  "$add_args"