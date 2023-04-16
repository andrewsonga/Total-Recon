gpu_id=$1
seqname=$2
export CXX=g++

# 1) extract reconstructed object meshes and reconstructed 6-DOF root-body / camera trajectories
bash extract_fgbg.sh $gpu_id $seqname

# 2) evalute method on stereo-view synthesis
bash scripts/render_nvs_fgbg_stereoview.sh $gpu_id $seqname
