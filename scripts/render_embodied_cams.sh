# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
add_args=${*: 3:$#-1}

testdir=logdir/$seqname   # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1               # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5

# Render Meshes for Reconstructed Objects, Egocentric Camera (Blue) and 3rd-Person-Follow Camera (Yellow)
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_embodied_cams.py \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $startframe \
  $add_args