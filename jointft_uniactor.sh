# prefix=human1-stereo000 or dog1-stereo000
prefix=$1
dep_wt=$2
gpus=$3
addr=$4
add_args=${*: 5:$#-1}

jointft_suffix=jointft
pretrain_suffix=e120-b256-ft2
sil_wt=0
flow_wt=1.0
ent_wt=0

# jointly finetuned model will be saved under logdir/$seqname
seqname=$prefix-leftcam-$jointft_suffix
loadname_fg=$prefix-leftcam-$pretrain_suffix
loadname_bkgd=$prefix-bkgd-leftcam-$pretrain_suffix

bash scripts/template-scene.sh $gpus $seqname $addr $dep_wt $sil_wt $flow_wt $ent_wt \
    --loadname_objs $loadname_fg --loadname_objs $loadname_bkgd \
    --learning_rate 0.0001 --nodense_trunc_eikonal_loss --nofreeze_root --ks_opt --nouse_ent --nouse_proj --nouse_embed \
    $add_args

# from run_scripts.sh
#./train_fgbg_ver2.sh $seqname 5.0 0.0 1.0 0.0 --loadname_objs $loadname_obj0 --loadname_objs $loadname_obj1 --loadname_objs $loadname_obj2 --use_unc --learning_rate 0.0001

# from ablation_objmotion_se3_nofreezecam
#./train_animal_fgbg_ver2.sh $seqname 5.0 0. 1.0 0.0 --loadname_objs $loadname_obj0 --loadname_objs $loadname_obj1 --loadname_objs $loadname_obj2 --learning_rate 0.0001 --nodense_trunc_eikonal_loss --nofreeze_root --ks_opt --nouse_ent --nouse_proj --nouse_embed

# from ablation_freezecam
#./train_animal_fgbg.sh $seqname 5.0 0. 1.0 0.0 --loadname_objs $loadname_obj0 --loadname_objs $loadname_obj1 --loadname_objs $loadname_obj2 --learning_rate 0.0001 --nodense_trunc_eikonal_loss --nofreeze_root --ks_opt --nouse_ent --nouse_proj --nouse_embed