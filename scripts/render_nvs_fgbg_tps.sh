# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
vidid=$3   # pose traj
rootid=$4  # root traj
add_args=${*: 5:$#-1}
#fg_normalbase_vertex_index=$5
#fg_downdir_vertex_index=$6

testdir=logdir/$seqname # %: from end
save_prefix=$testdir/nvs
startframe=0
maxframe=-1          # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5
sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

thirdpersoncam_offset_y=0
thirdpersoncam_offset_z=-0.15

#1st person view synthesis
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
  --thirdperson_view \
  $add_args
  
  #--thirdpersoncam_offset_y $thirdpersoncam_offset_y \
  #--thirdpersoncam_offset_z $thirdpersoncam_offset_z
  # offset_z = 0.05 is required so as to place camera in front of the haze in the vacinity of the surface

#--fg_normalbase_vertex_index $fg_normalbase_vertex_index \
#--fg_downdir_vertex_index $fg_downdir_vertex_index \

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