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
use_human=$4
use_symm=$5
dep_wt=$6
sil_wt=$7
flow_wt=${8}
ent_wt=${9}
num_epochs=120
batch_size=256

#loadname_fg=$3
#loadname_bkgd=$4

model_prefix=$seqname
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path
##########################
## 1st training stage
##########################
# generate opts.log for fg (obj0)
savename=${seqname}-init               # where to store obj0/, obj1, tensorboard files, rendered images
loadname_obj0=${savename}/obj0         # where to store the logs, models, vars
bash scripts/gen-optslog-mgpu.sh $gpus $loadname_obj0 $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2
#--log_mesh \

# generate opts.log for bkgd (obj1)
loadname_obj1=${savename}/obj1
bash scripts/gen-optslog-mgpu.sh $gpus $loadname_obj1 $savename \
    $seqname $addr --num_epochs $num_epochs \
  --use_rtk_file \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2 \
  --nolbs \
  --recon_bkgd \
  --proj_wt 0 \
  --sil_wt 0 \
  --alpha 6 \
  --bound_factor 20 \
  --nf_reset 0.1 \
  --bound_reset 0.1 \
  --noloss_flt --norm_novp --nolbs --nouse_proj --nouse_unc --dist_corresp --nounc_filter --noroot_sm --nouse_embed --anneal_freq # removing --nouse_embed flag to allow for eikonal loss computation without any errors

# run main_objs.py
bash scripts/template-objs-mgpu.sh $gpus $addr --loadname_objs $loadname_obj0 --loadname_objs $loadname_obj1
$loadnames

##########################
## 2nd training stage
##########################
# generate opts.log for fg (obj0)

# generate opts.log for bkgd (obj1)

# run main_objs.py
#loadnames=''
#bash scripts/template-objs-mgpu.sh $gpus $addr $loadnames

##########################
## 3rd training stage
##########################
# generate opts.log for fg (obj0)

# generate opts.log for bkgd (obj1)

# run main_objs.py
#loadnames=''
#bash scripts/template-objs-mgpu.sh $gpus $addr $loadnames