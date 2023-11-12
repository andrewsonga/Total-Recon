# prefix=human1-mono000 or dog1-mono000
prefix=$1
dep_wt=$2
gpus=$3         # e.g. 0,1,2,3
use_human=$4    # "" (for human) or "no" (for non-human)
addr=$5         # e.g. 10001
add_args=${*: 6:$#-1}

eikonal_wt=0.001

# pretrain foreground actor
seqname=$prefix
bash scripts/template-fg.sh $gpus $seqname $addr "$use_human" "no" $dep_wt $add_args

# pretrain bkgd object
# NOTE that for background pretraining: dep_wt=1
seqname_bkgd=$prefix-bkgd
bash scripts/template-bkgd.sh $gpus $seqname_bkgd $addr "$use_human" "no" 1.0 $eikonal_wt $add_args