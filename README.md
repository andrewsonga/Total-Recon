# Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis
#### [**Project**](https://andrewsonga.github.io/totalrecon/) | [**Paper**](https://arxiv.org/abs/2304.12317) 

This is the official PyTorch implementation of "Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis". 

<a href="https://andrewsonga.github.io">Chonghyuk Song</a>, <a href="https://gengshan-y.github.io">Gengshan Yang</a>, <a href="https://dunbar12138.github.io">Kangle Deng</a>, <a href="https://www.cs.cmu.edu/~junyanz/">Jun-Yan Zhu</a>, <a href="https://www.cs.cmu.edu/~deva/">Deva Ramanan</a>
<br>
Carnegie Mellon University
<br>
ICCV 2023

https://user-images.githubusercontent.com/20153928/232976346-581c6080-6451-4306-bbf2-e9a34aea1599.mp4

Given a long video of deformable objects captured by a handheld RGBD sensor, Total-Recon renders the scene from novel camera trajectories derived from in-scene motion of actors: (1) egocentric cameras that simulate the point-of-view of a target actor (such as the pet) and (2) 3rd-person (or pet) cameras that follow the actor from behind. Our method also enables (3) 3D video filters that attach virtual 3D assets to the actor. Total-Recon achieves this by reconstructing the geometry, appearance, and root-body and articulated motion of each deformable object in the scene as well as the background.

## Timeline
We plan to release our code in the following 4 stages:

- [x] Inference and Evaluation code for 4 select sequences of our stereo RGBD dataset
- [x] Raw data and pre-optimized models for all sequences of our dataset
- [x] Training code (per-object pretraining and joint-finetuning)
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

(1) Download and untar the raw data:
```
bash download_rawdata.sh

# untar raw data
tar -xzvf totalrecon_rawdata.tar.gz
```

(2) Appropriately relocate the raw data:
```
# place raw data under raw/
# argv[1]: The directory inside Total-Recon where the downloaded raw data is stored

src_dir=totalrecon_rawdata
bash place_rawdata.sh $src_dir
```

(3) Download the pre-trained VCN optical flow model for data preprocessing (instructions are taken from [BANMo](https://github.com/facebookresearch/banmo/tree/main/preprocess#download-optical-flow-model)):
```
mkdir lasr_vcn
wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./lasr_vcn/vcn_rob.pth
```

(4) Preprocess raw data (takes around a couple of hours per sequence):

Multi-actor sequences (e.g. human-dog):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).
# argv[2]: gpu number (0, 1, 2, ...)

prefix=humandog-stereo000; gpu=0

bash preprocess_maskcamgiven_rawdata_multiactor.sh $prefix $gpu
```

Uni-actor sequences (e.g. cat2):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).
# argv[2]: human or not, where `y` denotes human and  `n` denotes quadreped.
# argv[3]: gpu number (0, 1, 2, ...)

prefix=cat2-stereo000; ishuman='n'; gpu=0

bash preprocess_maskcamgiven_rawdata_uniactor.sh $prefix $ishuman $gpu
```

(5) [NOT REQUIRED FOR INFERENCE] Format preprocessed data for training:

Multi-actor sequences (e.g. human-dog):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).
# argv[2]: gpu number (0, 1, 2, ...)

prefix=humandog-stereo000; gpu=0

bash format_processeddata_multiactor.sh $prefix $gpu
```

Uni-actor sequences (e.g. cat2):
```
# argv[1]: Sequence name that points to folders under `raw/` (minus the suffix -leftcam or -rightcam).
# argv[2]: gpu number (0, 1, 2, ...)

prefix=cat2-stereo000; gpu=0

bash format_processeddata_uniactor.sh $prefix $gpu
```

### Pre-optimized Models
(1) Download the pre-optimized models and untar them.
```
bash download_models.sh

tar -xzvf totalrecon_models.tar.gz
```

(2) Appropriately relocate the pre-optimized models:
```
# Place the pre-optimized models under logdir/
# argv[1]: The directory inside Total-Recon where the downloaded pre-optimized models are stored

src_dir=totalrecon_models
bash place_models.sh $src_dir
```

### 3D Assets
To run the 3D video filter and to be able to visualize flying embodied-view cameras, purchase and download 3D models in .obj format for 1) [the unicorn horn](https://www.turbosquid.com/3d-models/3d-unicorn-horn-1510343), and 2) [the Canon camera](https://www.cgtrader.com/3d-models/electronics/video/canon-eos-5d-mark-iii-876c89b9-350b-45fc-8420-9d4df6471e65). 

Rename the .obj file for the camera mesh to `camera.obj`, then place the file `camera.obj` and unzipped folder `UnicornHorn_OBJ` inside `mesh_material`.

## Training

Run per-object pretraining and joint-finetuning as follows: 

```
# change appropriately (e.g. "humancat-stereo000" or "cat2-stereo000")
prefix=humandog-stereo000

bash train_$prefix.sh
```

## Inference

### Mesh and Root-body Pose Extraction
Before inference or evaluation can be done, extract the object-level meshes and root-body poses from the trained model. This only needs to be done once per model:
```
# argv[1]: gpu number (0, 1, 2, ...)
# argv[2]: folder name of the trained model inside logdir/

seqname=humandog-stereo000-leftcam-jointft    # (appropriately rename `seqname`)
bash extract_fgbg.sh $gpu $seqname
```

### Left-Right Camera Registration
Before inference or evaluation can be done, copy the left-right camera registration data from the raw data directory to the trained model directory:
```
prefix=humandog-stereo000                     # (appropriately rename `prefix`)
seqname=$prefix-leftcam-jointft               # directory name of trained model

# for uniactor sequences
cp raw/$prefix-leftcam/normrefcam2secondcam.npy logdir/$seqname/   

# for multiactor sequences     
cp raw/$prefix-human-leftcam/normrefcam2secondcam.npy logdir/$seqname/
```

### Egocentric View Synthesis

https://user-images.githubusercontent.com/20153928/234135753-610bc744-789e-4174-9b75-f7c979376506.mp4

(takes around a few hours)
The rendered videos will be saved as `nvs-fpsview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_fps.sh $gpu $seqname "$add_args"
```

<details><summary>per-sequence arguments <code>(add_args)</code></summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800 --fg_downdir_vertex_index 1874 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --asset_offset_z -0.05 --scale_fps 0.50"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 150324 --fg_downdir_vertex_index 150506 --asset_scale 0.003 --firstpersoncam_offset_y 0.05 --firstpersoncam_offsetabt_xaxis 25 --firstpersoncam_offsetabt_yaxis 15 --firstpersoncam_offsetabt_zaxis 5 --fix_frame 50 --scale_fps 0.75"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 204713 --fg_downdir_vertex_index 204830 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --scale_fps 0.75"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 34175 --fg_downdir_vertex_index 6043 --asset_scale 0.003 --firstpersoncam_offset_z 0.13 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis -15 --scale_fps 0.75"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 338844 --fg_downdir_vertex_index 166318 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 10 --firstpersoncam_offsetabt_yaxis 10 --asset_offset_z -0.05 --scale_fps 0.75"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 308732 --fg_downdir_vertex_index 309449 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 20 --firstpersoncam_offsetabt_zaxis -20 --scale_fps 0.75"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 105919 --fg_downdir_vertex_index 246367 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --scale_fps 0.75"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --asset_offset_z -0.05 --scale_fps 0.75"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 227642 --fg_downdir_vertex_index 117789 --asset_scale 0.003 --firstpersoncam_offset_z 0.035 --firstpersoncam_offsetabt_xaxis 45 --scale_fps 0.75"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 161978 --fg_downdir_vertex_index 37496 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 10 --firstpersoncam_offsetabt_yaxis 10 --asset_offset_z -0.05 --scale_fps 0.75"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.003 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_yaxis 20 --asset_offset_z -0.05 --scale_fps 0.75"
```

</details>
<br>


### 3rd-Person-Follow (3rd-Pet-Follow) View Synthesis

https://user-images.githubusercontent.com/20153928/234136211-02241af4-9e7b-486e-a2d3-36c21f384ecc.mp4

(takes around a few hours)
The rendered videos will be saved as `nvs-tpsview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_tps.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments <code>(add_args)</code></summary>

1) Human-dog 

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --asset_scale 0.003 --scale_tps 0.70"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_fgmeshcenter_elevate_y 1.00 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.45 --thirdpersoncam_offsetabt_yaxis 10 --asset_scale 0.003 --fix_frame 50 --scale_tps 0.70"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--thirdpersoncam_offset_x 0.25 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.38 --thirdpersoncam_offsetabt_zaxis 20 --asset_scale 0.003 --scale_tps 0.70"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--thirdpersoncam_offset_x -0.20 --thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.50 --thirdpersoncam_offsetabt_xaxis -10 --thirdpersoncam_offsetabt_yaxis 15 --thirdpersoncam_offsetabt_zaxis -25 --asset_scale 0.003 --scale_tps 0.70"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_zaxis 30 --asset_scale 0.003"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --scale_tps 0.70"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--thirdpersoncam_offset_x 0.10 --thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_yaxis 17 --thirdpersoncam_offsetabt_zaxis 25 --asset_scale 0.003 --scale_tps 0.70"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --asset_scale 0.003 --scale_tps 0.70"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_xaxis -10 --asset_scale 0.003 --scale_tps 0.70"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.13 --thirdpersoncam_offset_z -0.17 --asset_scale 0.003 --scale_tps 0.70"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --asset_scale 0.003 --scale_tps 0.70"
```

</details>
<br>

### Bird's-Eye View Synthesis

https://user-images.githubusercontent.com/20153928/234136114-e4b29bde-db35-466d-bde5-a9592dc6f341.mp4

(takes around a few hours)
The rendered videos will be saved as `nvs-bev-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_bev.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="add_args="--fg_obj_index 0 --fix_frame 157 --topdowncam_offset_x 0.01 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 150"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 470 --topdowncam_offset_x 0.15 --topdowncam_offset_y 0.40 --topdowncam_offsetabt_zaxis -210 --topdowncam_offsetabt_yaxis 30"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 130 --topdowncam_offset_x -0.30 --topdowncam_offset_y 0.55 --topdowncam_offset_z -0.18 --topdowncam_offsetabt_zaxis 4 --topdowncam_offsetabt_yaxis -22 --topdowncam_offsetabt_xaxis 10"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 14 --topdowncam_offset_x 0.40 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 185 --topdowncam_offsetabt_yaxis 30"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 110 --topdowncam_offset_x -0.3 --topdowncam_offset_y 0.80 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -7"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 440 --topdowncam_offset_x 0.35 --topdowncam_offset_y 0.70 --topdowncam_offset_z 0.1 --topdowncam_offsetabt_yaxis 20 --topdowncam_offsetabt_zaxis 20"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 18 --topdowncam_offset_y 0.90 --topdowncam_offset_z 0.30 --topdowncam_offsetabt_zaxis 10"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 50 --topdowncam_offset_x 0.1 --topdowncam_offset_y 0.95 --topdowncam_offset_z 0.32 --topdowncam_offsetabt_zaxis 23"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 120 --topdowncam_offset_x -0.17 --topdowncam_offset_y 0.80 --topdowncam_offset_z 0.25"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --fix_frame 40 --topdowncam_offset_x -0.070 --topdowncam_offset_y 0.28 --topdowncam_offset_z -0.03 --topdowncam_offsetabt_zaxis 30"
```

</details>
<br>

### Render 6-DoF Root-body Trajectory (Viewed from Bird's Eye View)

https://user-images.githubusercontent.com/20153928/234136515-e83aac52-92e7-45bc-83fa-320f5a9afecd.mp4

(takes around an hour)
The rendered video will be saved as `nvs-bev-traj-rootbody-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_rootbody --render_traj_bev $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 1 --fix_frame 65 --topdowncam_offset_x 0.10 --topdowncam_offset_y 0.60 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -15"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 157 --topdowncam_offset_x 0.01 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 150"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 470 --topdowncam_offset_x 0.15 --topdowncam_offset_y 0.40 --topdowncam_offsetabt_zaxis -210 --topdowncam_offsetabt_yaxis 30"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 130 --topdowncam_offset_x -0.30 --topdowncam_offset_y 0.55 --topdowncam_offset_z -0.18 --topdowncam_offsetabt_zaxis 4 --topdowncam_offsetabt_yaxis -22 --topdowncam_offsetabt_xaxis 10"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 14 --topdowncam_offset_x 0.40 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 185 --topdowncam_offsetabt_yaxis 30"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 110 --topdowncam_offset_x -0.3 --topdowncam_offset_y 0.80 --topdowncam_offset_z -0.05 --topdowncam_offsetabt_zaxis -7"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 440 --topdowncam_offset_x 0.35 --topdowncam_offset_y 0.70 --topdowncam_offset_z 0.1 --topdowncam_offsetabt_yaxis 20 --topdowncam_offsetabt_zaxis 20"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 18 --topdowncam_offset_y 0.90 --topdowncam_offset_z 0.30 --topdowncam_offsetabt_zaxis 10"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 50 --topdowncam_offset_x 0.1 --topdowncam_offset_y 0.95 --topdowncam_offset_z 0.32 --topdowncam_offsetabt_zaxis 23"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fix_frame 120 --fg_obj_index 0 --rootbody_obj_index 0 --topdowncam_offset_x -0.17 --topdowncam_offset_y 0.80 --topdowncam_offset_z 0.25"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --rootbody_obj_index 0 --fix_frame 40 --topdowncam_offset_x -0.070 --topdowncam_offset_y 0.28 --topdowncam_offset_z -0.03 --topdowncam_offsetabt_zaxis 30"
```

</details>
<br>

### Render 6-DoF Egocentric Camera Trajectory (Viewed from Stereo View)

https://user-images.githubusercontent.com/20153928/234138748-29303cd9-9876-417f-ac46-6553b73c5e3d.mp4

(takes around an hour)
The rendered video will be saved as `nvs-stereoview-traj-fpscam-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_fpscam --render_traj_stereoview $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog 

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --firstpersoncam_offsetabt_xaxis 15"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 150324 --fg_downdir_vertex_index 150506 --asset_scale 0.003 --firstpersoncam_offset_y 0.05 --firstpersoncam_offsetabt_xaxis 25 --firstpersoncam_offsetabt_yaxis 15 --firstpersoncam_offsetabt_zaxis 5 --fix_frame 50"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 204713 --fg_downdir_vertex_index 204830 --render_traj_stereoview --render_fpscam --firstpersoncam_offset_x -0.30 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis 10"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 34175 --fg_downdir_vertex_index 6043 --firstpersoncam_offset_z 0.13 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis -15 --fix_frame 0 --topdowncam_offset_y 0.60"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 338844 --fg_downdir_vertex_index 166318 --firstpersoncam_offsetabt_xaxis 10 --firstpersoncam_offsetabt_yaxis 10 --fix_frame 14 --topdowncam_offset_x 0.40 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 185 --topdowncam_offsetabt_yaxis -30"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 308732 --fg_downdir_vertex_index 309449 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 20 --firstpersoncam_offsetabt_zaxis -20 --asset_offset_z -0.05 --fix_frame 0 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_xaxis -40"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 105919 --fg_downdir_vertex_index 246367 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 15 --asset_offset_z -0.05 --fix_frame 30"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --fix_frame 18"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 227642 --fg_downdir_vertex_index 117789 --firstpersoncam_offset_z 0.035 --firstpersoncam_offsetabt_xaxis 45 --asset_offset_z -0.035 --fix_frame 50 --topdowncam_offset_y 0.60"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 161978 --fg_downdir_vertex_index 37496 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 10 --asset_offset_z -0.05"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80"
```

</details>
<br>

### Render 6-DoF 3rd-Person-Follow Camera Trajectory (Viewed from Stereo View)

https://user-images.githubusercontent.com/20153928/234137853-24000a96-32b6-4ad2-8d9c-bcee3639a129.mp4

(takes around an hour)
The rendered video will be saved as `nvs-stereoview-traj-tpscam-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_traj.sh $gpu $seqname --render_tpscam --render_traj_stereoview $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--rootbody_obj_index 1 --fg_obj_index 1 --asset_obj_index 1 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_fgmeshcenter_elevate_y 1.00 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.45 --thirdpersoncam_offsetabt_yaxis 10 --fix_frame 50"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x 0.25 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.38 --thirdpersoncam_offsetabt_zaxis 20"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x -0.20 --thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.50 --thirdpersoncam_offsetabt_xaxis -10 --thirdpersoncam_offsetabt_yaxis 15 --thirdpersoncam_offsetabt_zaxis -25 --fix_frame 0 --topdowncam_offset_y 0.60"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_zaxis 30 --fix_frame 14 --topdowncam_offset_x 0.40 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 185 --topdowncam_offsetabt_yaxis -30"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --fix_frame 0 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_xaxis -40"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x 0.10 --thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_yaxis 17 --thirdpersoncam_offsetabt_zaxis 25 --fix_frame 30"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --fix_frame 18"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_xaxis -10 --fix_frame 50 --topdowncam_offset_y 0.60"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.13 --thirdpersoncam_offset_z -0.17"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80"
```

</details>
<br>

### Render Meshes for Reconstructed Objects, Egocentric Camera (Blue), and 3rd-Person-Follow Camera (Yellow)

https://user-images.githubusercontent.com/20153928/234138995-1407d81d-7657-4bd9-8f52-757b852a379d.mp4

(takes around an hour)
The rendered video will be saved as `nvs-embodied-cams-mesh.mp4` inside `logdir/$seqname/`
```
bash scripts/render_embodied_cams.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800  --fg_downdir_vertex_index 1874 --asset_scale 0.003  --render_cam_stereoview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --asset_offset_z -0.05 --thirdpersoncam_offset_y 0.25 --thirdpersoncam_offset_z -0.80 --scale_fps 1.0 --scale_tps 1.0"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 150324 --fg_downdir_vertex_index 150506 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offset_y 0.05 --firstpersoncam_offsetabt_xaxis 25 --firstpersoncam_offsetabt_yaxis 15 --firstpersoncam_offsetabt_zaxis 5 --thirdpersoncam_fgmeshcenter_elevate_y 1.00 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.45 --thirdpersoncam_offsetabt_yaxis 10 --scale_fps 1.0 --scale_tps 1.0"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 204713 --fg_downdir_vertex_index 204830 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offset_x -0.30 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis 10 --scale_fps 1.0 --thirdpersoncam_offset_x 0.25 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.38 --thirdpersoncam_offsetabt_zaxis 20 --scale_tps 1.0"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 34175 --fg_downdir_vertex_index 6043 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offset_z 0.13 --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis -20 --firstpersoncam_offsetabt_zaxis -15 --thirdpersoncam_offset_x -0.20 --thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.50 --thirdpersoncam_offsetabt_xaxis -10 --thirdpersoncam_offsetabt_yaxis 15 --thirdpersoncam_offsetabt_zaxis -25 --scale_fps 1.0 --scale_tps 1.0"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 338844 --fg_downdir_vertex_index 166318 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offsetabt_xaxis 10 --firstpersoncam_offsetabt_yaxis 10 --thirdpersoncam_offset_y 0.15 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_zaxis 30 --scale_fps 1.0 --scale_tps 1.0 --fix_frame 14 --topdowncam_offset_x 0.40 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_zaxis 185 --topdowncam_offsetabt_yaxis -30"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 308732 --fg_downdir_vertex_index 309449 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 20 --firstpersoncam_offsetabt_zaxis -20 --asset_offset_z -0.05 --scale_fps 1.0 --thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --scale_tps 1.0 --fix_frame 0 --topdowncam_offset_y 0.60 --topdowncam_offsetabt_xaxis -40"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 105919 --fg_downdir_vertex_index 246367 --asset_scale 0.003 --render_cam_stereoview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 20 --firstpersoncam_offsetabt_yaxis 15 --asset_offset_z -0.05 --scale_fps 1.0 --thirdpersoncam_offset_x 0.10 --thirdpersoncam_offset_y 0.20 --thirdpersoncam_offset_z -0.60 --thirdpersoncam_offsetabt_yaxis 17 --thirdpersoncam_offsetabt_zaxis 25 --scale_tps 1.0 --fix_frame 30"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offsetabt_xaxis 35 --firstpersoncam_offsetabt_yaxis 30 --firstpersoncam_offsetabt_zaxis 20 --scale_fps 1.0 --thirdpersoncam_offset_x 0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.30 --thirdpersoncam_offset_y 0.50 --thirdpersoncam_offset_z -0.75 --thirdpersoncam_offsetabt_zaxis 20 --scale_tps 1.0 --fix_frame 18"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 227642 --fg_downdir_vertex_index 117789 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offset_z 0.035 --firstpersoncam_offsetabt_xaxis 45 --asset_offset_z -0.035 --scale_fps 1.0 --thirdpersoncam_offset_y 0.45 --thirdpersoncam_offset_z -0.80 --thirdpersoncam_offsetabt_xaxis -10 --scale_tps 1.0 --fix_frame 50 --topdowncam_offset_y 0.60"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 161978 --fg_downdir_vertex_index 37496 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offsetabt_xaxis 10 --firstpersoncam_offsetabt_yaxis 10 --scale_fps 1.0 --thirdpersoncam_fgmeshcenter_elevate_y 0.70 --thirdpersoncam_offset_y 0.13 --thirdpersoncam_offset_z -0.17 --scale_tps 1.0 --fix_frame 90 --topdowncam_offsetabt_yaxis 15 --topdowncam_offsetabt_xaxis -45"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.004 --render_cam_stereoview --firstpersoncam_offset_z 0.05 --firstpersoncam_offsetabt_xaxis 15 --firstpersoncam_offsetabt_zaxis 10 --asset_offset_z -0.05 --scale_fps 1.0 --thirdpersoncam_offset_x -0.05 --thirdpersoncam_fgmeshcenter_elevate_y 0.80 --thirdpersoncam_offset_y 0.05 --thirdpersoncam_offset_z -0.40 --scale_tps 1.0 --fix_frame 0 --topdowncam_offset_y 0.1 --topdowncam_offset_z 0.2 --topdowncam_offsetabt_yaxis -10 --topdowncam_offsetabt_xaxis -80"
```

</details>
<br>

### Render 3D Video Filters

https://user-images.githubusercontent.com/20153928/234139260-0f26370b-e0b3-4594-9a03-27635ad3d09c.mp4

(takes around a few hours)
The rendered video will be saved as `nvs-inputview-rgb_with_asset.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_3dfilter.sh $gpu $seqname $add_args
```

<details><summary>per-sequence arguments (<code>add_args</code>)</summary>

1) Human-dog

```
seqname=humandog-stereo000-leftcam-jointft

add_args="--fg_obj_index 1 --asset_obj_index 1 --fg_normalbase_vertex_index 96800 --fg_downdir_vertex_index 1874 --asset_scale 0.0006 --input_view --noevaluate"
```

2) Human-cat

```
seqname=humancat-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 150324 --fg_downdir_vertex_index 150506 --asset_scale 0.0006 --input_view --noevaluate"
```

3) Cat1 (v1)
```
seqname=cat1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 204713 --fg_downdir_vertex_index 204830 --asset_scale 0.0004 --asset_offsetabt_yaxis -50 --asset_offsetabt_zaxis 10 --asset_offset_x 0.10 --input_view --noevaluate"
```

4) Cat1 (v2)
```
seqname=cat1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 34175 --fg_downdir_vertex_index 6043 --asset_scale 0.0004 --input_view --noevaluate"
```

5) Cat2 (v1)
```
seqname=cat2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 338844 --fg_downdir_vertex_index 166318 --asset_scale 0.0004 --input_view --noevaluate"
```

6) Cat2 (v2)
```
seqname=cat2-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 308732 --fg_downdir_vertex_index 309449 --asset_scale 0.0003 --input_view --noevaluate"
```

7) Cat3
```
seqname=cat3-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 105919 --fg_downdir_vertex_index 246367 --asset_scale 0.0003 --asset_offsetabt_yaxis 15 --input_view --noevaluate"
```

8) Dog1 (v1)

```
seqname=dog1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 159244 --fg_downdir_vertex_index 93456 --asset_scale 0.0006 --asset_offsetabt_xaxis -25 --asset_offsetabt_yaxis 35 --input_view --noevaluate"
```

9) Dog1 (v2)
```
seqname=dog1-stereo001-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 227642 --fg_downdir_vertex_index 117789 --asset_scale 0.0005 --fix_frame 50 --topdowncam_offset_y 0.30 --input_view --noevaluate"
```

10) Human 1
```
seqname=human1-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 161978 --fg_downdir_vertex_index 37496 --asset_scale 0.0005 --asset_offsetabt_xaxis -20 --input_view --noevaluate"
```

11) Human 2

```
seqname=human2-stereo000-leftcam-jointft

add_args="--fg_obj_index 0 --asset_obj_index 0 --fg_normalbase_vertex_index 114756 --fg_downdir_vertex_index 114499 --asset_scale 0.0005 --asset_offsetabt_yaxis 10 --input_view --noevaluate"
```

</details>
<br>

## Evaluation

### Stereo View Synthesis (train on left camera, evaluate on right camera)

https://github.com/andrewsonga/Total-Recon/assets/20153928/9853a9a3-01e6-45f8-ad7c-d6c46b5d5d91

(takes around a few hours)
The rendered videos will be saved as `nvs-stereoview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_stereoview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view stereoview
```

### Train View Synthesis (train and evaluate on left camera)

https://user-images.githubusercontent.com/20153928/234143341-aab9935b-5604-4ca2-a568-874cae96ffec.mp4

(takes around a few hours)
The rendered videos will be saved as `nvs-inputview-*.mp4` inside `logdir/$seqname/`
```
bash scripts/render_nvs_fgbg_inputview.sh $gpu $seqname
python print_metrics.py --seqname $seqname --view inputview
```

---

## Citation

If you find this repository useful for your research, please cite the following work.
```
@inproceedings{song2023totalrecon,
  title={Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis},
  author={Song, Chonghyuk and Yang, Gengshan and 
          Deng, Kangle and Zhu, Jun-Yan and Ramanan, Deva},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

---

## Acknowledgements

We thank Nathaniel Chodosh, Jeff Tan, George Cazenavette, and Jason Zhang for proofreading our paper and Songwei Ge for reviewing our code. We also thank Sheng-Yu Wang, Daohan (Fred) Lu, Tamaki Kojima, Krishna Wadhwani, Takuya Narihira, and Tatsuo Fujiwara as well for providing valuable feedback. This work is supported in part by the Sony Corporation, Cisco Systems, Inc., and the CMU Argo AI Center for Autonomous Vehicle Research. This codebase is heavily based on [BANMo](https://github.com/facebookresearch/banmo) and uses evaluation code from [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields). The dataset was captured with the [Record3D](https://record3d.app) iOS app.
