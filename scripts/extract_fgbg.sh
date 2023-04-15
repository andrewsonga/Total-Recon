dev=$1
seqname=$2
add_args=${*: 3:$#-1}
vidid=0   # pose traj
rootid=0  # root traj

maxframe=-1       # setting this value to -1 means that we intend to use all frames of raw video (a positive integer value indicates we will either upsample or downsample the number frames)
scale=1
scale_rgbimages=1
sample_grid3d=256
add_args+=" --sample_grid3d ${sample_grid3d} --mc_threshold -0.002 --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

# extract meshes
CUDA_VISIBLE_DEVICES=$dev python extract_fgbg.py \
                  --seqname $seqname \
                  --test_frames {${rootid}","${vidid}} \
                  --dist_corresp \
                  $add_args