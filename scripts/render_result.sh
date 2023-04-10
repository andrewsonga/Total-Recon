# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
seqname=$1
model_path=$2
test_frames=$3

testdir=${model_path%/*} # %: from end
add_args=${*: 3:$#-1}
prefix=$testdir/$seqname-$test_frames

# part 1
python extract.py --flagfile=$testdir/opts.log \
                  --model_path $model_path \
                  --test_frames $test_frames \
                  $add_args
# freezing at first frame
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-frz \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --freeze \
#                     --vis_cam
# visualizing bones
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-bne \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp -1 \
                     --vis_bones \
#                     --vis_traj
# original camera view
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj0 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 0 \
#                     --vis_traj
# side view
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj1 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 1 \
#                     --vis_traj
# top view
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-trj2 \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --vp 2 \
#                     --vis_traj
# original video
python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $prefix-vid \
                     --seqname $seqname \
                     --test_frames $test_frames \
                     --append_img yes \
                     --append_render no
                     #--show_dp \

# part2 other renderings
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-cbne \
#                     --seqname $seqname \
#                     --test_frames $test_frames \
#                     --vp -2 \
#                     --vis_bones \
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-cgray \
#                     --seqname $seqname \
#                     --test_frames $test_frames \
#                     --vp -2 \
#                     --gray_color \
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-vbne \
#                     --seqname $seqname \
#                     --test_frames $test_frames \
#                     --vp 0 \
#                     --vis_bones \
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-vgray \
#                     --seqname $seqname \
#                     --test_frames $test_frames \
#                     --vp 0 \
#                     --gray_color
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-fgray \
#                     --seqname $seqname \
#                     --test_frames $test_frames \
#                     --vp -1 \
#                     --gray_color
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-rst \
#                     --seqname $seqname \
#                     --rest
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-err0 \
#                     --seqname $seqname \
#                     --vp 0  \
#                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-err1 \
#                     --seqname $seqname \
#                     --vp 1  \
#                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-err2 \
#                     --seqname $seqname \
#                     --vp 2  \
#                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
#python scripts/visualize/render_vis.py --testdir $testdir \
#                     --outpath $prefix-errs \
#                     --seqname $seqname \
#                     --vp -1 \
#                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/

# part 3
ffmpeg -y -i $prefix-vid.mp4 \
          -i $prefix-frz.mp4 \
          -i $prefix-bne.mp4 \
          -i $prefix-trj0.mp4 \
          -i $prefix-trj1.mp4 \
          -i $prefix-trj2.mp4 \
-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[top];\
[3:v][4:v][5:v]hstack=inputs=3[bottom];\
[top][bottom]vstack=inputs=2[v]" \
-map "[v]" \
$prefix-all.mp4
#-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[top];\
#[3:v][4:v][5:v]hstack=inputs=3[mid];\
#[6:v][7:v][8:v]hstack=inputs=3[bottom];\
#[top][mid][bottom]vstack=inputs=3[v]" \

ffmpeg -y -i $prefix-all.mp4 -vf "scale=iw/2:ih/2" $prefix-all.gif
#imgcat $prefix*.mp4
#imgcat $prefix-all.gif

####################################################################################################
################################### modified by Chonghyuk Song #####################################
# remove all of the intermediate images / videos to save space
rm $prefix-bne-*
rm $prefix-frz-*
rm $prefix-trj0-*
rm $prefix-trj1-*
rm $prefix-trj2-*

#rm $testdir/*.obj    # cat-pikachiu-rgbd%03d-bone-%05d.obj, cat-pikachiu-rgbd%03d-mesh-%05d.obj
#rm $testdir/*.jpg    # cat-pikachiu-rgbd%03d-flo-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-err-%05d.jpg, cat-pikachiu-rgbd%03d-img-gt-%05d.jpg, cat-pikachiu-rgbd%03d-img-p-%05d.jpg
####################################################################################################
####################################################################################################

#cp --parents $prefix-*.mp4 /data3/gengshay/banmo-vids/
#cp --parents $prefix-*.gif /data3/gengshay/banmo-vids/
