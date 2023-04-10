gpu_id=$1
seqname=$2
posetraj_id=$3
roottraj_id=$4      # the video from which we're gonna take the viewing direction trajectory
add_args=${*: 5:$#-1}
bash scripts/extract_fgbg.sh $gpu_id $seqname $posetraj_id $roottraj_id $add_args

#add_args may include
# --loadname_objs
# --savename_objs