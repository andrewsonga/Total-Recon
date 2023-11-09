# training script for user-provided, monocular RGBD videos (single-actor scenes)

gpus=$1                     # e.g. gpus=0,1,2,3 
addr=$2                     # e.g. addr=10001
prefix=$3                   # e.g. prefix=human2-mono000
use_human=$4                # "" (for human actors) / "no" (for animal actors)

# 1) hyperparameters
dep_wt=5.0                  # weight on depth loss
add_args="--lamb 0.9"       # interpolation factor for exponentially moving average update of object bounds and near-far plane

# 2) pretraining objects fields
bash pretrain_uniactor.sh $prefix $dep_wt $gpus "$use_human" $addr

# 3) joint finetuning
bash jointft_uniactor.sh $prefix $dep_wt $gpus $addr