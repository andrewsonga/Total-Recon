# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj
add_args=${*: 5:$#-1}

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=240
maxframe=242            # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5
sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

# WARNING: make sure when loading camera .txt files from folders containing pretrained models, only keep camera .txt files of the format $seqname-cam-%0d5.txt
## render (saves gt rgb, rendered rgb, sil, vis, dep: $nvs_outpath-rgbgt, $nvs_outpath-rgb, $nvs_outpath-sil, $nvs_outpath-vis, $nvs_outpath-dep
# 1. bev nvs

#(cat video)
#bev_frame=60
#(andrew video)
#bev_frame=55
#(andrew video 2)
bev_frame=130
topdowncam_offset_y=0.72
topdowncam_offset_x=-0.2
topdowncam_offset_z=0.08
#(cat amelie video)
#bev_frame=140
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
  --topdown_view \
  $add_args

: '
# combine rendered videos
ffmpeg -y -i $save_prefix-bev-rgbgt.mp4 \
          -i $save_prefix-bev-rgb.mp4 \
          -i $save_prefix-bev-dphgt.mp4 \
          -i $save_prefix-bev-dph.mp4 \
          -i $save_prefix-bev-mesh.mp4 \
          -i $save_prefix-bev-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-bev-all.mp4

ffmpeg -y -i $save_prefix-bev-all.mp4 -vf "scale=iw:ih" $save_prefix-bev-all.gif
'