gpus=$1                     # e.g. gpus=0,1,2,3 
addr=$2                     # e.g. addr=10001
prefix=dog1-stereo001

# 1) pretraining object fields
dep_wt=5.0
use_human="no"                # "" denotes human, "no" denotes quadreped (i.e. not-human)
bash pretrain_stereo_uniactor.sh $prefix $dep_wt $gpus "$use_human" $addr

# 2) joint finetuning
bash jointft_stereo_uniactor.sh $prefix $dep_wt $gpus $addr