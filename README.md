# Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis
#### [**Project**](https://andrewsonga.github.io/totalrecon/) | [**Paper**](https://andrewsonga.github.io/totalrecon/) 

This is the official PyTorch implementation of "Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis". 

<a href="https://andrewsonga.github.io">Chonghyuk Song</a>, <a href="https://gengshan-y.github.io">Gengshan Yang</a>, <a href="https://dunbar12138.github.io">Kangle Deng</a>, <a href="https://www.cs.cmu.edu/~junyanz/">Jun-Yan Zhu</a>, <a href="https://www.cs.cmu.edu/~deva/">Deva Ramanan</a>
<br>
Carnegie Mellon University
<br>
arXiv 2023

https://user-images.githubusercontent.com/20153928/232976346-581c6080-6451-4306-bbf2-e9a34aea1599.mp4

Given a long video of deformable objects captured by a handheld RGBD sensor, Total-Recon renders the scene from novel camera trajectories derived from in-scene motion of actors: (1) egocentric cameras that simulate the point-of-view of a target actor (such as the pet) and (2) 3rd-person (or pet) cameras that follow the actor from behind. Our method also enables (3) 3D video filters that attach virtual 3D assets to the actor. Total-Recon achieves this by reconstructing the geometry, appearance, and root-body and articulated motion of each deformable object in the scene as well as the background.

## Timeline
We plan to release our code in the following 4 stages:

- [x] Inference and Evaluation code for 4 select sequences of our stereo RGBD dataset
- [ ] Raw data for all sequences of our dataset
- [ ] Training code for all sequences of our dataset
- [ ] Data preprocessing code for user-provided RGBD videos

## Getting Started

### Dependencies

(1) Clone repo (including submodules):
```
git clone https://github.com/andrewsonga/Total-Recon.git --recursive

# This step is REQUIRED for all subsequent steps!
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
(4) Install ffmpeg:
```
apt-get install ffmpeg
```

### Data
We provide raw and preprocessed data for the "human-dog", "human-cat", "human2" and "dog1 (v1)" sequences for now, but we will release the raw data for all 11 sequences of our stereo RGBD dataset very soon.

(1) Download raw data and preprocessed data, and untar them.
```
bash download_rawdata.sh
bash download_preprocessed.sh

# untar raw data
tar -xzvf rawdata_forrelease.tar.gz

# untar preprocess data (approrpriately rename `filename`)
filename=database_humandog.tar.gz
tar -xzvf $filename
```

(2) Appropriately place the downloaded data with the following scripts:
```
# place raw data under raw/
# argv[1]: The directory inside Total-Recon where the downloaded raw data is stored

src_dir=rawdata_forrelease
bash place_rawdata.sh $src_dir

# place preprocessed data under database/ (approrpriately rename `src_dir`)
# argv[1]: The directory inside Total-Recon where the downloaded preprocessed data is stored

src_dir=database_humandog   
tgt_dir=database            
bash place_preprocessed.sh $src_dir $tgt_dir
```

(3) Download the pre-trained VCN optical flow model for data preprocessing (instructions are taken from [BANMo](https://github.com/facebookresearch/banmo/tree/main/preprocess#download-optical-flow-model)):
```
mkdir lasr_vcn
wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./lasr_vcn/vcn_rob.pth
```

(4) Preprocess raw data (don't run if you have already downloaded preprocessed data)

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
(1) Download the pre-trained models, and untar them.
```
bash download_models.sh

tar -xzvf pretrained_models_forrelease.tar.gz
```

(2) Appropriately place the downloaded pretrained models with the following script:
```
# Place the pre-trained models under logdir/
# argv[1]: The directory inside Total-Recon where the downloaded preprocessed data is stored

src_dir=pretrained_models_forrelease
bash place_models.sh $src_dir
```

### 3D Assets
To run the 3D video filter and to be able to visualize flying embodied-view cameras, purchase and download 3D models in .obj format for 1) [the unicorn horn](https://www.turbosquid.com/3d-models/3d-unicorn-horn-1510343), and 2) [the Canon camera](https://www.cgtrader.com/3d-models/electronics/video/canon-eos-5d-mark-iii-876c89b9-350b-45fc-8420-9d4df6471e65). 

Rename the .obj file for the camera mesh to `camera.obj`, then place the file `camera.obj` and unzipped folder `UnicornHorn_OBJ` inside `mesh_material`.

## Inference

### Mesh and Root-body Pose Extraction
Before inference or evaluation can be done, please extract the object-level meshes and root-body poses from the trained model:
```
# argv[1]: gpu id (0, 1, 2, ...)
# argv[2]: folder name of the trained model inside logdir/

seqname=humandog-stereo000-leftcam-jointft    # (appropriately rename `seqname`)
bash extract_fgbg.sh $gpu_id $seqname
```

### Egocentric View Synthesis
The rendered videos will be saved as `nvs-fpsview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_fps.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments <code>(add_args)</code></summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800 --fg_downdir_vertex_index 1874 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.50
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 170450 --fg_downdir_vertex_index 51716 --asset_scale 0.003 --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 0 --firstpersoncam_offsetabt_zaxis 0 --fix_frame 50 --scale_fps 0.75
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --asset_offset_z -0.05 --scale_fps 0.75
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_yaxis 20 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.75
```

</details>
<br>


### 3rd-Person-Follow (3rd-Pet-Follow) View Synthesis
The rendered videos will be saved as `nvs-tpsview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_tps.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments <code>(add_args)</code></summary>

1) Human-dog 

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_fgmeshcenter_elevate_y 0 --thirdpersoncam_offset_x 0 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_zaxis 0 --asset_scale 0.003 --scale_tps 0.70
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.16 --thirdpersoncam_offset_z -0.40 --asset_scale 0.003 --thirdpersoncam_offsetabt_zaxis 0 --thirdpersoncam_offsetabt_yaxis -10 --fix_frame 50 --scale_tps 0.70
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --thirdpersoncam_offsetabt_xaxis 0 --asset_scale 0.003 --scale_tps 0.70
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --thirdpersoncam_offsetabt_zaxis 0 --thirdpersoncam_offsetabt_xaxis 0 --asset_scale 0.003 --scale_tps 0.70
```

</details>
<br>

### Bird's-Eye View Synthesis
The rendered videos will be saved as `nvs-bev-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_bev.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --fix_frame 157 --topdowncam_offset_x 0.01 --topdowncam_offset_y 0.60 --topdowncam_offset_z 0 --topdowncam_offsetabt_zaxis 157
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --fix_frame 18 --topdowncam_offset_x -0.0 --topdowncam_offset_y 0.90 --topdowncam_offset_z 0.30 --topdowncam_offsetabt_zaxis 10 --topdowncam_offsetabt_yaxis 0
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --fix_frame 40 --topdowncam_offset_x -0.070 --topdowncam_offset_y 0.28 --topdowncam_offset_z -0.03 --topdowncam_offsetabt_zaxis 30 --topdowncam_offsetabt_yaxis 0
```

</details>
<br>

### Render 6-DoF Root-body Trajectory (Viewed from Bird's Eye View)
The rendered video will be saved as `nvs-bev-traj-rootbody-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_rootbody --render_traj_bev $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --rootbody_obj_index 1 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 157 --topdowncam_offset_x 0.01 --topdowncam_offset_y 0.60 --topdowncam_offset_z 0 --topdowncam_offsetabt_zaxis 157
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 18 --topdowncam_offset_x -0.0 --topdowncam_offset_y 0.90 --topdowncam_offset_z 0.30 --topdowncam_offsetabt_zaxis 10 --topdowncam_offsetabt_yaxis 0
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 40 --topdowncam_offset_x -0.070 --topdowncam_offset_y 0.28 --topdowncam_offset_z -0.03 --topdowncam_offsetabt_zaxis 30 --topdowncam_offsetabt_yaxis 0
```

</details>
<br>

### Render 6-DoF Egocentric Camera Trajectory (Viewed from Stereo View)
The rendered video will be saved as `nvs-stereoview-traj-fpscam-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_fpscam --render_traj_stereoview $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog 

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z 0
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 170450 --fg_downdir_vertex_index 51716 --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 0 --firstpersoncam_offsetabt_zaxis 0 --fix_frame 50
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --fix_frame 18 --topdowncam_offset_y 0.0 --topdowncam_offset_z 0 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis 0 --topdowncam_offsetabt_xaxis 0
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_yaxis 0 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80
```

</details>
<br>

### Render 6-DoF 3rd-Person-Follow Camera Trajectory (Viewed from Stereo View)
The rendered video will be saved as `nvs-stereoview-traj-tpscam-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_tpscam --render_traj_stereoview $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_fgmeshcenter_elevate_y 0 --thirdpersoncam_offset_x 0 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_zaxis 0
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.16 --thirdpersoncam_offset_z -0.40 --thirdpersoncam_offsetabt_zaxis 0 --thirdpersoncam_offsetabt_yaxis -10 --fix_frame 50
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --thirdpersoncam_offsetabt_xaxis 0 --fix_frame 18 --topdowncam_offset_y 0.0 --topdowncam_offset_z 0 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis 0 --topdowncam_offsetabt_xaxis 0
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --thirdpersoncam_offsetabt_zaxis 0 --thirdpersoncam_offsetabt_xaxis 0 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80
```

</details>
<br>

### Render Meshes for Reconstructed Objects, Egocentric Camera (Blue), and 3rd-Person-Follow Camera (Yellow)
The rendered video will be saved as `nvs-embodied-cams-mesh.mp4` inside `logdir/$seqname/`
```
bash scripts/render_embodied_cams.sh $gpu $seqname $render_view $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --asset_scale 0.003  --render_cam --render_cam_inputview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.50
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 170450 --fg_downdir_vertex_index 51716 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 0 --firstpersoncam_offsetabt_zaxis 0 --thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.16 --thirdpersoncam_offset_z -0.40 --thirdpersoncam_offsetabt_zaxis 0 --scale_fps 1.0 --scale_tps 1.0
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --scale_fps 1.0 --thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --thirdpersoncam_offsetabt_xaxis 0 --scale_tps 1.0 --fix_frame 18 --topdowncam_offset_y 0.0 --topdowncam_offset_z 0 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis 0 --topdowncam_offsetabt_xaxis 0
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_yaxis 0 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --scale_fps 1.0 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --thirdpersoncam_offsetabt_zaxis 0 --thirdpersoncam_offsetabt_xaxis 0 --scale_tps 1.0 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offset_x 0 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80
```

</details>
<br>

### Render 3D Video Filters
The rendered video will be saved as `nvs-inputview-rgb_with_asset.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_3dfilter.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args=--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800 --fg_downdir_vertex_index 1874 --asset_scale 0.0006 --input_view --noevaluate
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --input_view --asset_obj_index 0 --fg_normalbase_vertex_index 170450 --fg_downdir_vertex_index 51716 --asset_scale 0.0006 --input_view --noevaluate
```

3) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --input_view --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.0006 --input_view --asset_offsetabt_xaxis -25 --asset_offsetabt_yaxis 35 --noevaluate
```

4) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args=--fg_obj_index 0 --input_view --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.0005 --input_view --asset_offsetabt_yaxis 10 --noevaluate
```

</details>
<br>

## Evaluation

### Stereo View Synthesis (train on left camera, evaluate on right camera)
The rendered videos will be saved as `nvs-stereoview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_stereoview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view stereoview
```

### Train View Synthesis (train and evaluate on left camera)
The rendered videos will be saved as `nvs-inputview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_inputview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view inputview
```

## Acknowledgements

We thank Nathaniel Chodosh, Jeff Tan, George Cazenavette, and Jason Zhang for proofreading our paper and Songwei Ge for reviewing our code. We also thank Sheng-Yu Wang, Daohan (Fred) Lu, Tamaki Kojima, Krishna Wadhwani, Takuya Narihira, and Tatsuo Fujiwara as well for providing valuable feedback. This work is supported in part by the Sony Corporation and the CMU Argo AI Center for Autonomous Vehicle Research. This codebase is heavily based on [BANMo](https://github.com/facebookresearch/banmo) and also uses evaluation code from [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields).