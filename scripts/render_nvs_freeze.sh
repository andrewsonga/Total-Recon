# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
model_path=$3
vidid=$4   # pose traj
rootid=$5  # root traj
freeze_frame=$6

testdir=${model_path%/*} # %: from end
save_prefix=$testdir/nvs-$vidid-$rootid
maxframe=150
scale=1
scale_rgbimages=1
sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --full_mesh \
#  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"
add_args="--sample_grid3d ${sample_grid3d} --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

# extrat meshes
CUDA_VISIBLE_DEVICES=$dev python extract.py --flagfile=$testdir/opts.log \
                  --model_path $model_path \
                  --test_frames {${rootid}","${vidid}} \
                  $add_args

# re-render the trained sequence
prefix=$testdir/$seqname-{$vidid}
trgpath=$prefix-vgray

# raw video
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $trgpath-vid \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --append_img yes \
                     --append_render no \
                     --scale $scale

# masks
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $trgpath --vp 0 \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid} \
                     --gray_color \
                     --scale $scale \
                     --freeze \
                     --freeze_frame $freeze_frame
                     
                     #--outpath $prefix-fgray --vp -1 \ (was commented out in original repo)

## render
rootdir=$trgpath-ctrajs-
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs.py --flagfile=$testdir/opts.log \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0\
  --rootdir $rootdir --nvs_outpath $save_prefix-traj

# merge
echo $save_prefix-traj-rgb.mp4
ffmpeg -y -i $trgpath-vid.mp4 \
          -i $trgpath.mp4 \
          -i $save_prefix-traj-rgb.mp4 \
-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
-map "[v]" \
$save_prefix-traj-all.mp4

#ffmpeg -y -i $save_prefix-traj-all.mp4 -vf "scale=iw/2:ih/2" $save_prefix-traj-all.gif
ffmpeg -y -i $save_prefix-traj-all.mp4 -vf "scale=iw:ih" $save_prefix-traj-all.gif
