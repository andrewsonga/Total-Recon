# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Use precomputed root body poses
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
dep_wt=$6
eikonal_wt=$7
add_args=${*: 8:$#-1}
num_epochs=120
batch_size=256

###############################################################
################# modified by Chonghyuk Song ##################
model_prefix=$seqname-e$num_epochs-b$batch_size
###############################################################
###############################################################

# background reconstruction using known cameras
# 1. mode: line load
# difference from template-fg.sh
# 1) remove pose_net path flag
# 2) add use_rtk_file flag
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
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
  --nouse_cc \
  --disentangled_nerf \
  --alpha_sigma 6 \
  --num_freqs_sigma 6 \
  --mc_threshold -0.02 \
  --dense_trunc_eikonal_loss \
  --truncation 0.1 \
  --dense_trunc_eikonal_wt $eikonal_wt \
  --noloss_flt --norm_novp --nolbs --nouse_proj --nouse_unc --dist_corresp --nounc_filter --noroot_sm --nouse_embed --anneal_freq \
  "$add_args"
  # removing --nouse_embed flag to allow for eikonal loss computation without any errors

# 2. mode: fine tune with active+fine samples, large rgb loss wt and reset beta
loadname=${model_prefix}-init
savename=${model_prefix}-ft2
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --nouse_cc \
  --mc_threshold -0.02 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2 \
  --nolbs \
  --recon_bkgd \
  --proj_wt 0 \
  --sil_wt 0 \
  --alpha 10 \
  --disentangled_nerf \
  --alpha_sigma 6 \
  --num_freqs_sigma 6 \
  --dense_trunc_eikonal_loss \
  --truncation 0.1 \
  --dense_trunc_eikonal_wt $eikonal_wt \
  --noloss_flt --norm_novp --nolbs --nouse_proj --nouse_unc --dist_corresp --nounc_filter --noroot_sm --nouse_embed \
  "$add_args"
  # removing --nouse_embed flag to allow for eikonal loss computation without any errors