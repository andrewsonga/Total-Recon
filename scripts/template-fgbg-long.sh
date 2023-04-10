gpus=$1
seqname=$2
loadname_fg=$3
loadname_bkgd=$4
addr=$5
use_human=$6
use_symm=$7
dep_wt=$8
sil_wt=$9
flow_wt=${10}
ent_wt=${11}
num_epochs=120
batch_size=256

#savename is the name of folder where models are saved
#seqname is the name of the config file from which we get the datapath and the rtkpaths
model_prefix=$seqname-e$num_epochs-b$batch_size
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --lineload --batch_size $batch_size \
  --${use_symm}symm_shape \
  --${use_human}use_human \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 \
  --warmup_shape_ep 0 \
  --nowarmup_rootmlp \
  --loadname_fg $loadname_fg --loadname_bkgd $loadname_bkgd \
  --noanneal_freq \
  --alpha 10 \
  --use_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --feat_wt 0 --dist_corresp --use_ent --ent_wt $ent_wt   # for joint finetuning with 3d cycle loss and proj loss as reg. losses (no longer use the flag --sil_filter / --nosil_filter)
  #--use_unc 
  #--norm_novp

  #--freeze_coarse_minus_beta --freeze_body_mlp --use_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt                    # finetuning only camera parameters and sdf beta's with 3d cycle loss, proj. loss as the reg. losses
  #--freeze_coarse --freeze_body_mlp --freeze_shape --freeze_cvf --use_unc --dep_wt $dep_wt --sil_wt $sil_wt  --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt # finetuning only camera parameters with 3d cycle loss, proj. loss as the reg. losses
  #--nouse_proj --use_unc --dep_wt $dep_wt --sil_wt $sil_wt  --flow_wt $flow_wt --feat_wt 0 --eikonal_loss --dist_corresp --nosil_filter --norm_novp --use_ent --ent_wt $ent_wt   # for joint finetuning with 3d cycle loss as only reg. loss
  #--nouse_unc --dep_wt $dep_wt --sil_wt $sil_wt --flow_wt $flow_wt --dist_corresp --sil_filter --eikonal_loss                                                                    # for finetuning foreground
