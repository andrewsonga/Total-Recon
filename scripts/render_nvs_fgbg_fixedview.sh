# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj
add_args=${*: 5:$#-1}
flag_loadname='--loadname_objs'

counter=0
add_args_array=(${add_args[@]})
for item in $add_args_array
do
    counter=$((counter+1))
    if [[ $item == $flag_loadname ]]
    then
        loadname_obj=${add_args_array[$counter]}
        loadname_base=`dirname ${loadname_obj}`
        testdir=logdir/$loadname_base
    fi
    break
done

if [ -z "$testdir" ]
then
    testdir=logdir/$seqname # %: from end
fi
save_prefix=$testdir/nvs

startframe=0
maxframe=-1            # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5

# fixed-view nvs
# cat video
#fix_frame=0
# andrew video
fix_frame=240
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $startframe \
  --fixed_view \
  --fix_frame $fix_frame \
  $add_args

# combine rendered videos
ffmpeg -y -i $save_prefix-fixedview-rgbgt.mp4 \
          -i $save_prefix-fixedview-rgb.mp4 \
          -i $save_prefix-fixedview-dphgt.mp4 \
          -i $save_prefix-fixedview-dph.mp4 \
          -i $save_prefix-fixedview-mesh.mp4 \
          -i $save_prefix-fixedview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-fixedview-all.mp4

ffmpeg -y -i $save_prefix-fixedview-all.mp4 -vf "scale=iw:ih" $save_prefix-fixedview-all.gif