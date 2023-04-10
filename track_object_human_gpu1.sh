dev=$1
seqname=$2

testdir=logdir/$seqname          # %: from end
save_prefix=$testdir/objpose
nvs_save_prefix=$testdir/nvs
startframe=0
maxframe=-1                      # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=0.5

# andrew video
#fix_frame=0
#topdowncam_offset_y=0.7
#topdowncam_offset_x=-0.12       # +ve topdowncam_offset_x pushes the image leftwards in the image plane
#topdowncam_offset_z=0           # +ve topdowncam_offset_z pushes the image downwards in the image plane

# cat video
#fix_frame=250
#topdowncam_offset_y=0.5
#topdowncam_offset_x=0           # +ve topdowncam_offset_x pushes the image leftwards in the image plane
#topdowncam_offset_z=0           # +ve topdowncam_offset_z pushes the image downwards in the image plane

# human-dualrig-fgbg002
fix_frame=150
topdowncam_offset_y=0.75
topdowncam_offset_x=-0.12       # +ve topdowncam_offset_x pushes the image leftwards in the image plane
topdowncam_offset_z=0           # +ve topdowncam_offset_z pushes the image downwards in the image plane

: '
# visualize object and camera poses
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/track_object_gpu1.py \
                  --seqname $seqname \
                  --nvs_outpath $save_prefix \
                  --startframe $startframe \
                  --maxframe $maxframe \
                  --fix_frame $fix_frame \
                  --scale $scale_rgbimages \
                  --topdowncam_offset_y $topdowncam_offset_y \
                  --topdowncam_offset_x $topdowncam_offset_x \
                  --topdowncam_offset_z $topdowncam_offset_z
'
ffmpeg -y -i $nvs_save_prefix-inputview-rgbgt.mp4 \
          -i $save_prefix.mp4 \
-filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
-map "[v]" \
$save_prefix-all.mp4

ffmpeg -y -i $save_prefix-all.mp4 -vf "scale=iw:ih" $save_prefix-all.gif

