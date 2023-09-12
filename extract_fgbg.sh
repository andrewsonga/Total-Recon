# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

gpu_id=$1
seqname=$2
add_args=${*: 3:$#-1}
bash scripts/extract_fgbg.sh $gpu_id $seqname $add_args

#add_args may include
# --loadname_objs
# --savename_objs