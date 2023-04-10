gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
dep_wt=$6
sil_wt=$7
flow_wt=$8
ent_wt=$9
add_args=${*: 10:$#-1}
num_epochs=30
batch_size=256

#savename is the name of folder where models are saved
#seqname is the name of the config file from which we get the datapath and the rtkpaths
model_prefix=$seqname-e$num_epochs-b$batch_size
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 \
  --nofreeze_root --nofreeze_shape \
  --use_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --proj_wt 0 --feat_wt 0 --frnd_wt 0 --dist_corresp --norm_novp --use_ent --ent_wt $ent_wt \
  $add_args # for joint finetuning with 3d cycle loss and proj loss as reg. losses (no longer use the flag --sil_filter / --nosil_filter)

  #--${use_symm}symm_shape \
  #--${use_human}use_human \

  #--freeze_coarse_minus_beta --freeze_body_mlp --use_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt                    # finetuning only camera parameters and sdf beta's with 3d cycle loss, proj. loss as the reg. losses
  #--freeze_coarse --freeze_body_mlp --freeze_shape --freeze_cvf --use_unc --dep_wt $dep_wt --sil_wt $sil_wt  --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt # finetuning only camera parameters with 3d cycle loss, proj. loss as the reg. losses
  #--nouse_proj --use_unc --dep_wt $dep_wt --sil_wt $sil_wt  --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt   # for joint finetuning with 3d cycle loss as only reg. loss
  #--nouse_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --dist_corresp --sil_filter --eikonal_loss                                                                    # for finetuning foreground
