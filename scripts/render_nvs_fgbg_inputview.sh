# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
add_args=${*: 3:$#-1}
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
maxframe=-1                 # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5

# Training View Synthesis
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --use_3dcomposite \
  --nvs_outpath $save_prefix \
  --dist_corresp \
  --startframe $startframe \
  --input_view \
  $add_args

# Concatenate Rendered Videos
ffmpeg -y -i $save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix-inputview-rgb.mp4 \
          -i $save_prefix-inputview-rgbabsdiff.mp4 \
          -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
          -map "[v]" \
          $save_prefix-inputview-comparergb.mp4

ffmpeg -y -i $save_prefix-inputview-dphgt.mp4 \
          -i $save_prefix-inputview-dph.mp4 \
          -i $save_prefix-inputview-dphabsdiff.mp4 \
          -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
          -map "[v]" \
          $save_prefix-inputview-comparedph.mp4

ffmpeg -y -i $save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix-inputview-rgb.mp4 \
          -i $save_prefix-inputview-dphgt.mp4 \
          -i $save_prefix-inputview-dph.mp4 \
          -filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
          -map "[v]" \
          $save_prefix-inputview-all.mp4

# Convert .mp4 to .gif
ffmpeg -y -i $save_prefix-inputview-all.mp4 -vf "scale=iw:ih" $save_prefix-inputview-all.gif