# Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis
#### [**Project**](https://andrewsonga.github.io/totalrecon/) | [**Paper**](https://andrewsonga.github.io/totalrecon/) 

This is the official PyTorch implementation of "Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis". 

https://andrewsonga.github.io/totalrecon/my_figures/cover_figure.mp4

Given a long video of deformable objects captured by a handheld RGBD sensor, Total-Recon renders the scene from novel camera trajectories derived from in-scene motion of actors: (1) egocentric cameras that simulate the point-of-view of a target actor (such as the pet) and (2) 3rd-person (or pet) cameras that follow the actor from behind. Our method also enables (3) 3D video filters that attach virtual 3D assets to the actor. Total-Recon achieves this by reconstructing the geometry, appearance, and root-body and articulated motion of each deformable object in the scene as well as the background.

## Getting Started

### Dependencies

(1) Clone repo (including submodules):
```
git clone https://github.com/andrewsonga/Total-Recon.git --recursive
cd Total-Recon
```
(2) Install conda env:
```
conda env create -f misc/totalrecon-cu113.yml
conda activate totalrecon-cu113
```
(3) Install submodules:
```
pip install -e third_party/pytorch3d
pip install -e third_party/kmeans_pytorch
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

### Data
We provide preprocessed data for the "human-dog" and "human-cat" sequences, and the raw data for all sequences.

(1) Download data and optical flow model for data preprocessing:
```
mkdir ./lasr_vcn
gdown https://drive.google.com/uc?id=139S6pplPvMTB-_giI6V2dxpOHGqqAdHn -O ./lasr_vcn/vcn_rob.pth
```

(2) Appropriate place the downloaded data with the following scripts:
```
# place raw data under raw/

# place preprocessed data under database/DAVIS/
```

(3) Preprocess raw data

Multi-foreground-object sequences (e.g. humandog):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).

bash preprocess_rawdata_multiobj.sh humandog-stereo
```

Single-foreground-object sequences (e.g. cat2):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).
# argv[2]: human or not, where `y` denotes human and  `n` denotes quadreped.

bash preprocess_rawdata_singleobj.sh cat2-stereo n
```

### Pre-trained Models
Our pre-trained models for all sequences can be downloaded using the following script:
```
bash 
```

## Inference

#### Egocentric View Synthesis
```
bash scripts/render_nvs_fgbg_fps.sh $gpu $seqname $add_args
```

#### 3rd-Person-Follow (3rd-Pet-Follow) View Synthesis
```
bash scripts/render_nvs_fgbg_tps.sh $gpu $seqname $add_args
```

#### Bird's-Eye View Synthesis
```
bash scripts/render_nvs_fgbg_bev.sh $gpu $seqname $add_args
```

#### Stereo View Synthesis (train on left camera, evaluate on right camera)
```
bash scripts/render_nvs_fgbg_stereoview.sh $gpu $seqname $add_args
```

#### Train View Synthesis
```
bash scripts/render_nvs_fgbg_inputview.sh $gpu $seqname $add_args
```

#### Render 6-DoF Root-body Trajectory (Viewed from Bird's Eye View)
```
bash scripts/render_traj.sh $gpu $seqname --render_rootbody --render_traj_bev $add_args
```

#### Render 6-DoF Egocentric Camera Trajectory (Viewed from Stereo View)
```
bash scripts/render_traj.sh $gpu $seqname --render_fpscam --render_traj_stereoview $add_args
```

#### Render 6-DoF 3rd-Person-Follow Camera Trajectory (Viewed from Stereo View)
```
bash scripts/render_traj.sh $gpu $seqname --render_tpscam --render_traj_stereoview $add_args
```

#### Render Meshes for Reconstructed Objects, Egocentric Camera (Blue), and 3rd-Person-Follow Camera (Yellow)
```
bash scripts/render_embodied_cams.sh $gpu $seqname $render_view $add_args
```

#### Render 3D Video Filters
```
bash scripts/render_nvs_fgbg_3dfilter.sh $gpu $seqname $add_args
```