# prefix=human1-stereo000 or dog1-stereo000
prefix=$1
dep_wt=$2
gpus=$3         # e.g. 0,1,2,3
ishuman=$4      # y/n
addr=$5         # e.g. 10001
add_args=${*: 6:$#-1}

eikonal_wt=0.001

# pretrain foreground actor
seqname=$prefix-leftcam
bash scripts/template-fg.sh $gpus $seqname $addr $ishuman "no" $dep_wt $add_args

# pretrain bkgd object
# NOTE that for background pretraining: dep_wt=1
seqname=$prefix-bkgd-leftcam
bash scripts/template-bkgd.sh $gpus $seqname $addr $ishuman "no" 1.0 $eikonal_wt $add_args