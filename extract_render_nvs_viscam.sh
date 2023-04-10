seqname=$1
vidid=$2
echo seqname: $seqname
./extract_render.sh $vidid $seqname $vidid
./nvs.sh $vidid $seqname $vidid $vidid
./vis_cams.sh $seqname
