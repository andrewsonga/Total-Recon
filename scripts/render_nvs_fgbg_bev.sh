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
add_args=${*: 3:$#-1}

testdir=logdir/$seqname   # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1               # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5

# WARNING: make sure when loading camera .txt files from folders containing pretrained models, only keep camera .txt files of the format $seqname-cam-%0d5.txt
## render (saves gt rgb, rendered rgb, sil, vis, dep: $nvs_outpath-rgbgt, $nvs_outpath-rgb, $nvs_outpath-sil, $nvs_outpath-vis, $nvs_outpath-dep

# Bird's-Eye View Synthesis
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $startframe \
  --topdown_view \
  $add_args

# Concatenate Rendered Videos
ffmpeg -y -i $save_prefix-bev-rgbgt.mp4 \
          -i $save_prefix-bev-rgb.mp4 \
          -i $save_prefix-bev-dphgt.mp4 \
          -i $save_prefix-bev-dph.mp4 \
          -i $save_prefix-bev-mesh.mp4 \
          -i $save_prefix-bev-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-bev-all.mp4

# Convert .mp4 to .gif
ffmpeg -y -i $save_prefix-bev-all.mp4 -vf "scale=iw:ih" $save_prefix-bev-all.gif