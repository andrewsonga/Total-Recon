# prefix=humandog-stereo000 or humancat-stereo000
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
loadname_human=$prefix-human-leftcam-$pretrain_suffix
loadname_animal=$prefix-animal-leftcam-$pretrain_suffix
loadname_bkgd=$prefix-bkgd-leftcam-$pretrain_suffix

bash scripts/template-scene.sh $gpus $seqname $addr $dep_wt $sil_wt $flow_wt $ent_wt \
    --loadname_objs $loadname_human --loadname_objs $loadname_animal --loadname_objs $loadname_bkgd \
    --learning_rate 0.0001 --nodense_trunc_eikonal_loss --nofreeze_root --ks_opt --nouse_ent --nouse_proj --nouse_embed \
    $add_args