# prefix=humandog-mono000 or humancat-mono000
prefix=$1
dep_wt=$2
gpus=$3         # e.g. 0,1,2,3
addr=$4         # e.g. 10001
add_args=${*: 5:$#-1}

eikonal_wt=0.001

# pretrain human actor
seqname_human=$prefix-human
bash scripts/template-fg.sh $gpus $seqname_human $addr "" "no" $dep_wt $add_args

# pretrain animal actor
seqname_animal=$prefix-animal
bash scripts/template-fg.sh $gpus $seqname_animal $addr "no" "no" $dep_wt $add_args

# pretrain bkgd object
# NOTE that for the background pretraining in multi-actor setting: is_human=""
# and for background pretraining in both uni-actor and multi-actor settings: dep_wt=1
seqname_bkgd=$prefix-bkgd
bash scripts/template-bkgd.sh $gpus $seqname_bkgd $addr "" "no" 1.0 $eikonal_wt $add_args