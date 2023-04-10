gpu_num=$1
seqname=$2
video_ids=$3
bash scripts/render_mgpu.sh $gpu_num $seqname logdir/$seqname-e120-b256-ft3/params_latest.pth $video_ids 256
