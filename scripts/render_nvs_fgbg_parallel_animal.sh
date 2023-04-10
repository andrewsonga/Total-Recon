# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1             # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5
sample_grid3d=256

# remove any existing videos
#rm $testdir/*.mp4
#rm $testdir/*.gif

# WARNING: make sure when loading camera .txt files from folders containing pretrained models, only keep camera .txt files of the format $seqname-cam-%0d5.txt
## render (saves gt rgb, rendered rgb, sil, vis, dep: $nvs_outpath-rgbgt, $nvs_outpath-rgb, $nvs_outpath-sil, $nvs_outpath-vis, $nvs_outpath-dep
# 1. bev nvs

#(cat video)
#bev_frame=60
#(andrew video)
#bev_frame=55
#(cat amelie video)
bev_frame=140
topdowncam_offset_y=0.85
cmd4=`echo CUDA_VISIBLE_DEVICES=0 python scripts/visualize/nvs_fgbg.py \
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
  --fix_frame $bev_frame`
cmd5=`echo ffmpeg -y -i $save_prefix-bev-rgbgt.mp4 \
  -i $save_prefix-bev-rgb.mp4 \
  -i $save_prefix-bev-dphgt.mp4 \
  -i $save_prefix-bev-dph.mp4 \
  -i $save_prefix-bev-mesh.mp4 \
  -i $save_prefix-bev-sil-obj0.mp4 \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
  -map "[v]" \
  $save_prefix-bev-all.mp4`
cmd6=`echo ffmpeg -y -i $save_prefix-bev-all.mp4 -vf "scale=iw:ih" $save_prefix-bev-all.gif`
screen -dmS "t1" bash -c "$cmd4; $cmd5; $cmd6"

# 2. standard turntable-style nvs (followed by egocentric view synthesis)
freeze_frame=0
cmd1=`echo CUDA_VISIBLE_DEVICES=1 python scripts/visualize/nvs_fgbg.py \
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
  --maxframe 40`
cmd2=`echo ffmpeg -y -i $save_prefix-frozentime-rgbgt.mp4 \
          -i $save_prefix-frozentime-rgb.mp4 \
          -i $save_prefix-frozentime-dphgt.mp4 \
          -i $save_prefix-frozentime-dph.mp4 \
          -i $save_prefix-frozentime-mesh.mp4 \
          -i $save_prefix-frozentime-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-frozentime-all.mp4`
cmd3=`echo ffmpeg -y -i $save_prefix-frozentime-all.mp4 -vf "scale=iw:ih" $save_prefix-frozentime-all.gif`
#andrew video
#fps_startframe=0
#fps_maxframe=100
#cat video
#fps_startframe=260
#fps_maxframe=330
#cat amelie video
fps_startframe=0
fps_maxframe=-1
fg_normalbase_vertex_index=55
fg_downdir_vertex_index=302
firstpersoncam_offset_z=0.05
cmd4=`echo CUDA_VISIBLE_DEVICES=1 python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
  --seqname $seqname \
  --scale $scale_rgbimages --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $fps_startframe \
  --maxframe $fps_maxframe \
  --firstperson_view \
  --fg_normalbase_vertex_index $fg_normalbase_vertex_index \
  --fg_downdir_vertex_index $fg_downdir_vertex_index \
  --firstpersoncam_offset_z $firstpersoncam_offset_z`
cmd5=`echo ffmpeg -y -i $save_prefix-fpsview-rgbgt.mp4 \
          -i $save_prefix-fpsview-rgb.mp4 \
          -i $save_prefix-fpsview-dphgt.mp4 \
          -i $save_prefix-fpsview-dph.mp4 \
          -i $save_prefix-fpsview-mesh.mp4 \
          -i $save_prefix-fpsview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-fpsview-all.mp4`
cmd6=`echo ffmpeg -y -i $save_prefix-fpsview-all.mp4 -vf "scale=iw:ih" $save_prefix-fpsview-all.gif`
screen -dmS "t2" bash -c "$cmd1; $cmd2; $cmd3; $cmd4; $cmd5; $cmd6"

# with depth supervision
#--firstpersoncam_offset_y 0.05 \
#--firstpersoncam_offset_z 0.05

# without depth supervision
#--firstpersoncam_offset_y 0.075 \
#--firstpersoncam_offset_z 0.075

# 3. input-view reconstruction (followed by third-person view synthesis)
cmd1=`echo CUDA_VISIBLE_DEVICES=2 python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
  --seqname $seqname \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --input_view \
  --startframe $startframe`
cmd2=`echo ffmpeg -y -i $save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix-inputview-rgb.mp4 \
          -i $save_prefix-inputview-dphgt.mp4 \
          -i $save_prefix-inputview-dph.mp4 \
          -i $save_prefix-inputview-mesh.mp4 \
          -i $save_prefix-inputview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-inputview-all.mp4`
cmd3=`echo ffmpeg -y -i $save_prefix-inputview-all.mp4 -vf "scale=iw:ih" $save_prefix-inputview-all.gif`
#andrew video
#tps_startframe=0
#tps_maxframe=100
#cat video
#tps_startframe=260
#tps_maxframe=330
#cat amelie video
tps_startframe=0
tps_maxframe=-1
cmd4=`echo CUDA_VISIBLE_DEVICES=2 python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
  --seqname $seqname \
  --scale $scale_rgbimages --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 \
  --nvs_outpath $save_prefix \
  --use_3dcomposite \
  --dist_corresp \
  --startframe $tps_startframe \
  --maxframe $tps_maxframe \
  --thirdperson_view \
  --thirdpersoncam_offset_y 0. \
  --thirdpersoncam_offset_z -0.25`
cmd5=`echo ffmpeg -y -i $save_prefix-tpsview-rgbgt.mp4 \
          -i $save_prefix-tpsview-rgb.mp4 \
          -i $save_prefix-tpsview-dphgt.mp4 \
          -i $save_prefix-tpsview-dph.mp4 \
          -i $save_prefix-tpsview-mesh.mp4 \
          -i $save_prefix-tpsview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-tpsview-all.mp4`
cmd6=`echo ffmpeg -y -i $save_prefix-tpsview-all.mp4 -vf "scale=iw:ih" $save_prefix-tpsview-all.gif`
screen -dmS "t3" bash -c "$cmd1; $cmd2; $cmd3; $cmd4; $cmd5; $cmd6"

# 4. stereo-view nvs (followed by fixed-view nvs)
# cat amelie video
fix_frame=400
# cat video
#fix_frame=250
#fix_frame=0    (we set fix_frame=0 for maxframe=100, and fix_frame=250 for maxframe=-1)
# andrew video
#fix_frame=20
cmd1=`echo CUDA_VISIBLE_DEVICES=3 python scripts/visualize/nvs_fgbg.py \
  --vidid $vidid \
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
  --refcam leftcam`
cmd2=`echo ffmpeg -y -i $save_prefix-stereoview-rgbgt.mp4 \
  -i $save_prefix-stereoview-rgbgt-secondcam.mp4 \
  -i $save_prefix-stereoview-rgb.mp4 \
  -i $save_prefix-stereoview-dphgt-secondcam.mp4 \
  -i $save_prefix-stereoview-dph.mp4 \
  -i $save_prefix-stereoview-mesh.mp4 \
  -i $save_prefix-stereoview-sil-obj0.mp4 \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]hstack=inputs=7[v]" \
  -map "[v]" \
  $save_prefix-stereoview-all.mp4`
cmd3=`echo ffmpeg -y -i $save_prefix-stereoview-all.mp4 -vf "scale=iw:ih" $save_prefix-stereoview-all.gif`
cmd4=`echo CUDA_VISIBLE_DEVICES=3 python scripts/visualize/nvs_fgbg.py \
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
  --fix_frame $fix_frame`
cmd5=`echo ffmpeg -y -i $save_prefix-fixedview-rgbgt.mp4 \
          -i $save_prefix-fixedview-rgb.mp4 \
          -i $save_prefix-fixedview-dphgt.mp4 \
          -i $save_prefix-fixedview-dph.mp4 \
          -i $save_prefix-fixedview-mesh.mp4 \
          -i $save_prefix-fixedview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-fixedview-all.mp4`
cmd6=`echo ffmpeg -y -i $save_prefix-fixedview-all.mp4 -vf "scale=iw:ih" $save_prefix-fixedview-all.gif`
screen -dmS "t4" bash -c "$cmd1; $cmd2; $cmd3; $cmd4; $cmd5; $cmd6"

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
