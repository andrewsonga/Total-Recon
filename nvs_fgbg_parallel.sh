gpu_id=$1
seqname=$2
posetraj_id=$3
roottraj_id=$4      # the video from which we're gonna take the viewing direction trajectory
bash scripts/render_nvs_fgbg_parallel.sh $gpu_id $seqname $posetraj_id $roottraj_id