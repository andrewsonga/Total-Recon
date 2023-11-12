gpus=$1                     # e.g. gpus=0,1,2,3
addr=$2                     # e.g. addr=10001
prefix=humancat-stereo000

# 1) pretraining object fields
dep_wt=5.0
add_args="--lamb 0.9"       # [OPTIONAL] updates object bounds and near-far plane via exponentially moving average, with an interpolation factor of lambda = 0.9. The camera-ready reports metrics for lambda = 0.0, but we found that this model may fail to converge from time to time; setting lambda = 0.9 stabilizes training without deviating from the reported metrics.
bash scripts/pretrain_stereo_multiactor.sh $prefix $dep_wt $gpus $addr $add_args

# 2) joint finetuning
bash scripts/jointft_stereo_multiactor.sh $prefix $dep_wt $gpus $addr
