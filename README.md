# Total-Recon: Deformable Scene Reconstruction for Embodied View Synthesis
#### [[Webpage]](https://andrewsonga.github.io/totalrecon/) 

<!--[[Arxiv]](https://arxiv.org/abs/2112.12761)-->

## Getting Started 

### Dependencies

```
```

### Data

```
```

### Pre-trained Models

```
```

### Inference

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