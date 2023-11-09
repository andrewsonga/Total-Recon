# training script for user-provided, monocular RGBD videos (multi-actor scenes)

gpus=$1                     # e.g. gpus=0,1,2,3 
addr=$2                     # e.g. addr=10001
prefix=$3                   # e.g. prefix=humandog-mono000

# 1) hyperparameters
dep_wt=5.0                  # weight on depth loss
add_args="--lamb 0.9"       # interpolation factor for exponentially moving average update of object bounds and near-far plane

# 2) pretraining object fields
bash pretrain_multiactor.sh $prefix $dep_wt $gpus $addr $add_args

# 3) joint finetuning
bash jointft_multiactor.sh $prefix $dep_wt $gpus $addr