# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
model_path=$3
vidid=$4   # pose traj
rootid=$5  # root traj

testdir=${model_path%/*} # %: from end
save_prefix=$testdir/nvs-$vidid-$rootid
maxframe=-1       # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
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

# testdir = '%s/%s-cam-%05d.txt'%(args.testdir, seqname_root, fr_root)
# e.g. testdir = logdir/$seqname-e120-b256-ft3/cat-pikachiu-rgbd-bkgd000-cam00000.txt

# raw video (saves frames of the raw rgb video: $seqname-{vidid}-vgray-vid.gif, as well as reference silhouette and camera trajectory: $seqname-{vidid}-vgray-refsil-%05d.png, $seqname-{vidid}-vgray-ctrajs-%05d.txt)
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $trgpath-vid \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --append_img yes \
                     --append_render no \
                     --scale $scale

# masks (saves renderings of the mesh: $seqname-{vidid}-vgray.gif) and camera trajectory: $seqname-{vidid}-vgray-ctrajs-%05d.txt)
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $trgpath --vp 0 \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid} \
                     --gray_color \
                     --scale $scale
                     #--rest

                     #--outpath $prefix-fgray --vp -1 \ (was commented out in original repo)

## render (saves rendered rgb, sil, and vis: nvs-$vidid-$rootid-traj-rgb, nvs-$vidid-$rootid-traj-sil, nvs-$vidid-$rootid-traj-vis)
rootdir=$trgpath-ctrajs-
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs.py --flagfile=$testdir/opts.log \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale_rgbimages --maxframe $maxframe --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0\
  --rootdir $rootdir --nvs_outpath $save_prefix-traj

####################################################################################################
################################### modified by Chonghyuk Song #####################################
# merge
#echo $save_prefix-traj-rgb.mp4
#ffmpeg -y -i $trgpath-vid.mp4 \
#          -i $trgpath.mp4 \
#          -i $save_prefix-traj-rgb.mp4 \
#-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
#-map "[v]" \
#$save_prefix-traj-all.mp4

echo $save_prefix-traj-rgb.mp4
ffmpeg -y -i $trgpath-vid.mp4 \
          -i $trgpath.mp4 \
          -i $save_prefix-traj-sil.mp4 \
          -i $save_prefix-traj-rgb.mp4 \
-filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[v]" \
-map "[v]" \
$save_prefix-traj-all.mp4
####################################################################################################
####################################################################################################

#ffmpeg -y -i $save_prefix-traj-all.mp4 -vf "scale=iw/2:ih/2" $save_prefix-traj-all.gif
ffmpeg -y -i $save_prefix-traj-all.mp4 -vf "scale=iw:ih" $save_prefix-traj-all.gif

####################################################################################################
################################### modified by Chonghyuk Song #####################################
# remove all of the intermediate images / videos to save space
rm $save_prefix-traj-rgb*
rm $save_prefix-traj-sil*
rm $save_prefix-traj-vis*
rm $trgpath*

#rm $testdir/*.obj    # cat-pikachiu-rgbd%03d-bone-%05d.obj, cat-pikachiu-rgbd%03d-mesh-%05d.obj
#rm $testdir/*.jpg    # cat-pikachiu-rgbd%03d-flo-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-gt-%05d.jpg, cat-pikachiu-rgbd%03d-img-p-%05d.jpg
####################################################################################################
####################################################################################################