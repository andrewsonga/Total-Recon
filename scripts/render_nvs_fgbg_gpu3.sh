# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=10
maxframe=11            # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
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

tps_startframe=20
tps_maxframe=50
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_fgbg.py \
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
  --thirdpersoncam_offset_y 0.15 \
  --thirdpersoncam_offset_z -0.1
ffmpeg -y -i $save_prefix-tpsview-rgbgt.mp4 \
          -i $save_prefix-tpsview-rgb.mp4 \
          -i $save_prefix-tpsview-dphgt.mp4 \
          -i $save_prefix-tpsview-dph.mp4 \
          -i $save_prefix-tpsview-mesh.mp4 \
          -i $save_prefix-tpsview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-tpsview-all.mp4
ffmpeg -y -i $save_prefix-tpsview-all.mp4 -vf "scale=iw:ih" $save_prefix-tpsview-all.gif
'

# 3. 1st person view reconstruction (followed by third-person view synthesis)
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
  --firstperson_view \
  --fg_normalbase_vertex_index 55 \
  --fg_downdir_vertex_index 302 \
  --firstpersoncam_offset_z 0.05 \
  --firstpersoncam_offset_y -0.20
  # offset_z = 0.05 is required so as to place camera in front of the haze in the vacinity of the surface

# (for human video)
#--fg_normalbase_vertex_index 111713 \
#--fg_downdir_vertex_index 111433 \
#--firstpersoncam_offset_z 0.05 
# --asset_scale 0.0004 

# (for catamelie video)
# --fg_normalbase_vertex_index 55 \
# --fg_downdir_vertex_index 302 \
#--firstpersoncam_offset_z 0.05  
# --asset_scale 0.0003

# (for cat pikachiu)
# --fg_normalbase_vertex_index 10102 \  # REMEMBER THAT SELECTING VERTICES VIA MESHLAB IS SUPER NOISY e.g. you may think youre choosing a vertex on a given side of the mesh, but the program may penetrate the mesh and choose a vertex on the side of the mesh.
# --asset_scale 0.0003

ffmpeg -y -i $save_prefix-fpsview-rgbgt.mp4 \
          -i $save_prefix-fpsview-rgb.mp4 \
          -i $save_prefix-fpsview-dphgt.mp4 \
          -i $save_prefix-fpsview-dph.mp4 \
          -i $save_prefix-fpsview-mesh.mp4 \
          -i $save_prefix-fpsview-sil-obj0.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v]hstack=inputs=6[v]" \
-map "[v]" \
$save_prefix-fpsview-all.mp4

ffmpeg -y -i $save_prefix-fpsview-all.mp4 -vf "scale=iw:ih" $save_prefix-fpsview-all.gif

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
