# prefix=human1-mono000 or dog1-mono000
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
seqname=$prefix-$jointft_suffix
loadname_fg=$prefix-$pretrain_suffix
loadname_bkgd=$prefix-bkgd-$pretrain_suffix

bash scripts/template-scene.sh $gpus $seqname $addr $dep_wt $sil_wt $flow_wt $ent_wt \
    --loadname_objs $loadname_fg --loadname_objs $loadname_bkgd \
    --learning_rate 0.0001 --nodense_trunc_eikonal_loss --nofreeze_root --ks_opt --nouse_ent --nouse_proj --nouse_embed \
    $add_args