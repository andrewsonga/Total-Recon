dev=$1
seqname=$2
model_path=$3
testdir=${model_path%/*} # %: from end
save_prefix=$testdir/framesim-

CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/framesim.py --flagfile=$testdir/opts.log \
  --model_path $model_path 