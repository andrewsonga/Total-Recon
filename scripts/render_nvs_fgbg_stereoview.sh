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

# Stereo View Synthesis
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $startframe \
  --stereo_view \
  --noflow_correspondence \
  --refcam leftcam \
  $add_args

# Concatenate Rendered Videos
ffmpeg -y -i $save_prefix-stereoview-rgbgt-secondcam.mp4 \
          -i $save_prefix-stereoview-rgb.mp4 \
          -i $save_prefix-stereoview-rgbabsdiff.mp4 \
          -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
          -map "[v]" \
          $save_prefix-stereoview-comparergb.mp4

ffmpeg -y -i $save_prefix-stereoview-dphgt-secondcam.mp4 \
          -i $save_prefix-stereoview-dph.mp4 \
          -i $save_prefix-stereoview-dphabsdiff.mp4 \
          -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
          -map "[v]" \
          $save_prefix-stereoview-comparedph.mp4

ffmpeg -y -i $save_prefix-stereoview-rgbgt.mp4 \
          -i $save_prefix-stereoview-rgbgt-secondcam.mp4 \
          -i $save_prefix-stereoview-rgb.mp4 \
          -i $save_prefix-stereoview-dphgt-secondcam.mp4 \
          -i $save_prefix-stereoview-dph.mp4 \
          -filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
          -map "[v]" \
          $save_prefix-stereoview-all.mp4

# Convert .mp4 to .gif
ffmpeg -y -i $save_prefix-stereoview-all.mp4 -vf "scale=iw:ih" $save_prefix-stereoview-all.gif