# prefix=humandog-stereo000 or humancat-stereo000
prefix=$1
dep_wt=$2
gpus=$3         # e.g. 0,1,2,3
addr=$4         # e.g. 10001
add_args=${*: 5:$#-1}

eikonal_wt=0.001

# pretrain human actor
seqname=$prefix-human-leftcam
bash scripts/template-fg.sh $gpus $seqname $addr "" "no" $dep_wt $add_args

# pretrain animal actor
seqname=$prefix-animal-leftcam
bash scripts/template-fg.sh $gpus $seqname $addr "no" "no" $dep_wt $add_args

# pretrain bkgd object
# NOTE that for the background pretraining in multi-actor setting: is_human=""
# and for background pretraining in both uni-actor and multi-actor settings: dep_wt=1
seqname=$prefix-bkgd-leftcam
bash scripts/template-bkgd.sh $gpus $seqname $addr "" "no" 1.0 $eikonal_wt $add_args