# Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis
#### [**Project**](https://andrewsonga.github.io/totalrecon/) | [**Paper**](https://andrewsonga.github.io/totalrecon/) 

This is the official PyTorch implementation of "Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis". 



https://user-images.githubusercontent.com/20153928/232976346-581c6080-6451-4306-bbf2-e9a34aea1599.mp4

Given a long video of deformable objects captured by a handheld RGBD sensor, Total-Recon renders the scene from novel camera trajectories derived from in-scene motion of actors: (1) egocentric cameras that simulate the point-of-view of a target actor (such as the pet) and (2) 3rd-person (or pet) cameras that follow the actor from behind. Our method also enables (3) 3D video filters that attach virtual 3D assets to the actor. Total-Recon achieves this by reconstructing the geometry, appearance, and root-body and articulated motion of each deformable object in the scene as well as the background.

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
We provide raw and preprocessed data for the "human-dog", "human-cat", "human2" and "dog1 (v1)" sequences.

(1) Download raw data and preprocessed data, and untar them.
```
bash download_rawdata.sh
bash download_preprocess.sh

tar -xzvf $filename
```

(2) Appropriately place the downloaded data with the following scripts:
```
# place raw data under raw/
# argv[1]: The directory inside Total-Recon where the downloaded raw data is stored

src_dir=rawdata_forrelease
bash place_rawdata.sh $src_dir

# place preprocessed data under database/
# argv[1]: The directory inside Total-Recon where the downloaded preprocessed data is stored

# for e.g.
src_dir=database_humandog             
bash place_preprocessed.sh $src_dir
```

(3) Download the optical flow model for data preprocessing:
```
mkdir lasr_vcn
gdown https://drive.google.com/uc?id=139S6pplPvMTB-_giI6V2dxpOHGqqAdHn -O lasr_vcn/vcn_rob.pth
```

(4) Preprocess raw data

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

tar -xzvf $filename
```

(2) Appropriately place the downloaded pretrained models with the following script:
```
# Place the pre-trained models under logdir/
# argv[1]: The directory inside Total-Recon where the downloaded preprocessed data is stored

src_dir=pretrained_models_forrelease
bash place_models.sh $src_dir
```

## Inference

#### Egocentric View Synthesis
```
bash scripts/render_nvs_fgbg_fps.sh $gpu $seqname $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --asset_scale 0.003  --render_cam --render_cam_inputview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.50
```

</details>
<br>


#### 3rd-Person-Follow (3rd-Pet-Follow) View Synthesis
```
bash scripts/render_nvs_fgbg_tps.sh $gpu $seqname $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_fgmeshcenter_elevate_y 0 --thirdpersoncam_offset_x 0 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_zaxis 0 --asset_scale 0.003 --scale_tps 0.70
```

</details>
<br>

#### Bird's-Eye View Synthesis
```
bash scripts/render_nvs_fgbg_bev.sh $gpu $seqname $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 0 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15
```

</details>
<br>

#### Render 6-DoF Root-body Trajectory (Viewed from Bird's Eye View)
```
bash scripts/render_traj.sh $gpu $seqname --render_rootbody --render_traj_bev $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 0 --rootbody_obj_index 1 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15
```

</details>
<br>

#### Render 6-DoF Egocentric Camera Trajectory (Viewed from Stereo View)
```
bash scripts/render_traj.sh $gpu $seqname --render_fpscam --render_traj_stereoview $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --firstpersoncam_offset_z 0 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z 0
```

</details>
<br>

#### Render 6-DoF 3rd-Person-Follow Camera Trajectory (Viewed from Stereo View)
```
bash scripts/render_traj.sh $gpu $seqname --render_tpscam --render_traj_stereoview $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_fgmeshcenter_elevate_y 0 --thirdpersoncam_offset_x 0 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_zaxis 0
```

</details>
<br>

#### Render Meshes for Reconstructed Objects, Egocentric Camera (Blue), and 3rd-Person-Follow Camera (Yellow)
```
bash scripts/render_embodied_cams.sh $gpu $seqname $render_view $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --asset_scale 0.003  --render_cam --render_cam_inputview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.50
```

</details>
<br>

#### Render 3D Video Filters
```
bash scripts/render_nvs_fgbg_3dfilter.sh $gpu $seqname $add_args
```
<details><summary>additional arguments (add_args)</summary>
Human-dog (seqname=humandog-stereo000-leftcam-jointft)

```
--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --asset_scale 0.003  --render_cam --render_cam_inputview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 0 --asset_offset_z -0.05 --scale_fps 0.50
```

</details>
<br>

## Evaluation

#### Stereo View Synthesis (train on left camera, evaluate on right camera)
```
bash scripts/render_nvs_fgbg_stereoview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view stereoview
```

#### Train View Synthesis (train and evaluate on left camera)
```
bash scripts/render_nvs_fgbg_inputview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view inputview
```
