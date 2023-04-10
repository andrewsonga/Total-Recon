# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3

testdir=logdir/$seqname         # %: from end
save_prefix=$testdir/
startframe=0
maxframe=1                     # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.375
sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --full_mesh \
#  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"
add_args="--sample_grid3d ${sample_grid3d} --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/depth2points.py \
  --seqname $seqname \
  --vidid $vidid \
  --scale $scale_rgbimages --maxframe $maxframe  --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $startframe

# combine rendered videos
ffmpeg -y -i ${save_prefix}dph-obj0.mp4 \
          -i ${save_prefix}dphgt-obj0.mp4 \
          -i ${save_prefix}dph-obj1.mp4 \
          -i ${save_prefix}dphgt-obj1.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
-map "[v]" \
${save_prefix}dph-all.mp4

ffmpeg -y -i ${save_prefix}dph-all.mp4 -vf "scale=iw:ih" ${save_prefix}dph-all.gif