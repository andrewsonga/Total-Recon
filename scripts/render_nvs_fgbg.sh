# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1            # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5
sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --full_mesh \
#  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"
add_args="--sample_grid3d ${sample_grid3d} --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

: '
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe 280 \
  --maxframe 335 \
  --thirdperson_view

# combine rendered videos
ffmpeg -y -i $save_prefix-tpsview-rgbgt.mp4 \
          -i $save_prefix-tpsview-rgb.mp4 \
          -i $save_prefix-tpsview-dph.mp4 \
          -i $save_prefix-tpsview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
-map "[v]" \
$save_prefix-tpsview-all.mp4

ffmpeg -y -i $save_prefix-tpsview-all.mp4 -vf "scale=iw:ih" $save_prefix-tpsview-all.gif
'

# WARNING: make sure when loading camera .txt files from folders containing pretrained models, only keep camera .txt files of the format $seqname-cam-%0d5.txt
## render (saves gt rgb, rendered rgb, sil, vis, dep: $nvs_outpath-rgbgt, $nvs_outpath-rgb, $nvs_outpath-sil, $nvs_outpath-vis, $nvs_outpath-dep
# 1. bev nvs

#(cat video)
#bev_frame=60
#(andrew video)
#bev_frame=55
#(andrew video 2)
bev_frame=130
topdowncam_offset_y=0.42
topdowncam_offset_x=-0.2
topdowncam_offset_z=0.08
#(cat amelie video)
#bev_frame=140
: '
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
  --topdowncam_offset_y $topdowncam_offset_y \
  --topdowncam_offset_x $topdowncam_offset_x \
  --topdowncam_offset_z $topdowncam_offset_z \
  --fix_frame $bev_frame
'
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

: '
# 2. standard turntable-style nvs
freeze_frame=0
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
  --freeze \
  --freeze_frame $freeze_frame \
  --endangle -50 \
  --startangle 30 \
  --maxframe 40

# combine rendered videos
ffmpeg -y -i $save_prefix-frozentime-rgbgt.mp4 \
          -i $save_prefix-frozentime-rgb.mp4 \
          -i $save_prefix-frozentime-dph.mp4 \
          -i $save_prefix-frozentime-mesh.mp4 \
          -i $save_prefix-frozentime-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
-map "[v]" \
$save_prefix-frozentime-all.mp4

ffmpeg -y -i $save_prefix-frozentime-all.mp4 -vf "scale=iw:ih" $save_prefix-frozentime-all.gif

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
  --startframe $startframe

# combine rendered videos
ffmpeg -y -i $save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix-inputview-rgb.mp4 \
          -i $save_prefix-inputview-dph.mp4 \
          -i $save_prefix-inputview-mesh.mp4 \
          -i $save_prefix-inputview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
-map "[v]" \
$save_prefix-inputview-all.mp4

ffmpeg -y -i $save_prefix-inputview-all.mp4 -vf "scale=iw:ih" $save_prefix-inputview-all.gif

# 4. fixed-view nvs
# cat video
#fix_frame=0
# andrew video
fix_frame=20
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
  --fix_frame $fix_frame

# combine rendered videos
ffmpeg -y -i $save_prefix-fixedview-rgbgt.mp4 \
          -i $save_prefix-fixedview-rgb.mp4 \
          -i $save_prefix-fixedview-dph.mp4 \
          -i $save_prefix-fixedview-mesh.mp4 \
          -i $save_prefix-fixedview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
-map "[v]" \
$save_prefix-fixedview-all.mp4

ffmpeg -y -i $save_prefix-fixedview-all.mp4 -vf "scale=iw:ih" $save_prefix-fixedview-all.gif

####################################################################################################
################################### modified by Chonghyuk Song #####################################
# remove all of the intermediate images / videos to save space
#rm $save_prefix-traj-rgb*
#rm $save_prefix-traj-sil*
#rm $save_prefix-traj-vis*
#rm $trgpath*

#rm $testdir/*.obj    # cat-pikachiu-rgbd%03d-bone-%05d.obj, cat-pikachiu-rgbd%03d-mesh-%05d.obj
#rm $testdir/*.jpg    # cat-pikachiu-rgbd%03d-flo-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-gt-%05d.jpg, cat-pikachiu-rgbd%03d-img-p-%05d.jpg
####################################################################################################
####################################################################################################
'