# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
dep_wt=$6
add_args=${*: 7:$#-1}
num_epochs=120
batch_size=256

model_prefix=$seqname-e$num_epochs-b$batch_size

# mode: line load
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2 \
  --log_mesh \
  "$add_args"

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
loadname=${model_prefix}-init
savename=${model_prefix}-ft1
num_epochs=$((num_epochs/4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1\
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2 \
  --log_mesh \
  "$add_args"

# mode: fine tune with active+fine samples, large rgb loss wt and reset beta
loadname=${model_prefix}-ft1
savename=${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --dep_wt $dep_wt \
  --dep_scale 0.2 \
  --log_mesh \
  "$add_args"
