# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

dev=$1
testdir=$2

# T_swing
gtdir=database/T_swing/meshes/
gt_pmat=database/T_swing/calibration/Camera1.Pmat.cal
seqname=ama-female
seqname_eval=T_swing1

## T_samba
#gtdir=database/T_samba/meshes/
#gt_pmat=database/T_samba/calibration/Camera1.Pmat.cal
#seqname=ama-female
#seqname_eval=T_samba1

## eagle
#gtdir=database/DAVIS/Meshes/Full-Resolution/a-eagle-1/
#gt_pmat=canonical
#seqname=a-eagle
#seqname_eval=a-eagle-1

## hands
#gtdir=database/DAVIS/Meshes/Full-Resolution/a-hands-1/
#gt_pmat=canonical
#seqname=a-hands
#seqname_eval=a-hands-1

# this part is not needed it meshes are already extracted
# model_path=$testdir/params_latest.pth
#sample_grid3d=256
#add_args="--sample_grid3d ${sample_grid3d} --full_mesh \
#  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"
#
#testdir=${model_path%/*} # %: from end
#CUDA_VISIBLE_DEVICES=$dev python extract.py --flagfile=$testdir/opts.log \
#                  --model_path $model_path \
#                  --test_frames {0} \
#                  $add_args

# evaluation
outfile=`cut -d/ -f2 <<<"${testdir}"`
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-pred \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} \
 > tmp/$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-gt \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --vis_gtmesh

# save to videos
ffmpeg -y -i $testdir/$seqname-eval-gt.mp4 \
          -i $testdir/$seqname-eval-pred.mp4 \
-filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
-map "[v]" \
$testdir/$seqname-all.mp4
