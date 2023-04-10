# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj
add_args=${*: 5:$#-1}

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1              # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5
sample_grid3d=256

# 3. input-view reconstruction
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
  --filter_3d \
  $add_args

#--input_view \

: '
ffmpeg -y -i $save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix-inputview-rgb.mp4 \
          -i $save_prefix-inputview-dphgt.mp4 \
          -i $save_prefix-inputview-dph.mp4 \
          -i $save_prefix-inputview-mesh.mp4 \
          -i $save_prefix-inputview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-inputview-all.mp4
ffmpeg -y -i $save_prefix-inputview-all.mp4 -vf "scale=iw:ih" $save_prefix-inputview-all.gif
'