gpus=$1                     # e.g. gpus=0,1,2,3 
addr=$2                     # e.g. addr=10001
prefix=cat1-stereo001

# 1) pretraining object fields
dep_wt=5.0
ishuman="no"                # "" denotes human, "no" denotes quadreped (i.e. not-human)
bash pretrain_uniactor.sh $prefix $dep_wt $gpus "$ishuman" $addr

# 2) joint finetuning
bash jointft_uniactor.sh $prefix $dep_wt $gpus $addr --nofreeze_shape_bkgd