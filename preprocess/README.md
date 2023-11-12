## Preprocess and format Record3D RGBD videos

Here we provide separate instructions for preprocessing and formatting *your own* RGBD videos. These instructions make the following assumptions about your data:

<details><summary> <b>Data assumptions</b></summary>

- The user-provided RGBD videos are captured with the [Record3D](https://record3d.app) iOS app.
- These videos are *monocular* sequences, unlike the *stereo* sequences provided in Total-Recon's dataset.
- The object masks are *not* provided by the user, and the camera parameters are *yet* to be formatted into the desired OpenCV format.
- For videos containing multi-actor scenes, the actors belong to different categories (human-pet is ok but human-human or pet-pet is not ok). This is because the current codebase uses semantic categories to track and mask an object. Using a video instance segmentation algorithm, such as [Track Anything](https://github.com/gaomingqi/Track-Anything), would enable one to remove this restriction.
</details>



### 1) Export Record3D data
- Export the RGBD video captured by the Record3D app (in `r3d` format). This will result in a folder that is named by a timestamp (`yyyy-mm-dd--hh-mm-ss`) and that contains a subdirectory named "Shareable" where the data is saved as a `.r3d` file.
- Save that folder (`yyyy-mm-dd--hh-mm-ss`) under `./raw` 

### 2) Unzip, crop, and subsample Record3D data
Unzip the `.r3d` file, and temporally crop and subsample the raw Record3D data with the following command:
- if no cropping is desired, set `start_frame=0` and `end_frame=-1`
- if no subsampling is desired, set `tgt_fps=$src_fps`, where `src_fps` is the frame rate of the raw Record3D video
```
# e.g. 
src_seqname="2023-11-11--00-00-00"
tgt_seqname=human2-mono000
start_frame=10
end_frame=-1
src_fps=30
tgt_fps=10

bash preprocess/unzip_crop_subsample.sh $src_seqname $tgt_seqname $start_frame $end_frame $src_fps $tgt_fps

###############################################################
# argv[1]: name of source directory exported by Record3D
# argv[2]: name of target directory in Total-Recon/$rootdir (will contain temporally cropped and subsampled RGBD sequence) 
# argv[3]: desired first frame in the raw Record3D sequence
# argv[4]: desired end frame (inclusive) in the raw Record3D sequence
# argv[5]: frame rate of the raw Record3D sequence
# argv[6]: desired frame rate for subsampling the raw Record3D sequence
```

### 3) Preprocess raw data (takes around an hour per sequence)

Multi-actor sequences:
```
# e.g.
prefix=humancat-mono000; gpu=0
bash preprocess/preprocess_rawdata_multiactor.sh $prefix $gpu

###############################################################
# argv[1]: prefix of the preprocessed data folders under "database/DAVIS/JPEGImages/" (minus suffices such as "-leftcam", "-rightcam", "-human", "-animal", "-bkgd", and "-uncropped")
# argv[2]: gpu number (0, 1, 2, ...)
```

Uni-actor sequences:
```
# e.g.
prefix=human2-mono000; ishuman='y'; gpu=0
bash preprocess/preprocess_rawdata_uniactor.sh $prefix $ishuman $gpu

###############################################################
# argv[1]: prefix of the preprocessed data folders under "database/DAVIS/JPEGImages/" (minus suffices such as "-leftcam", "-rightcam", and "-bkgd")
# argv[2]: human or not, where `y` denotes human and  `n` denotes quadreped
# argv[3]: gpu number (0, 1, 2, ...)
```

### 4) [NOT REQUIRED FOR INFERENCE] Format preprocessed data for training

Multi-actor sequences:
```
# e.g.
prefix=humancat-mono000; gpu=0
bash preprocess/format_processeddata_multiactor.sh $prefix $gpu

###############################################################
# argv[1]: prefix of the preprocessed data folders under "database/DAVIS/JPEGImages/" (minus suffices such as "-leftcam", "-rightcam", "-human", "-animal", "-bkgd", and "-uncropped")
# argv[2]: gpu number (0, 1, 2, ...)
```

Uni-actor sequences:
```
# e.g.
prefix=human2-mono000; gpu=0
bash preprocess/format_processeddata_uniactor.sh $prefix $gpu

###############################################################
# argv[1]: prefix of the preprocessed data folders under "database/DAVIS/JPEGImages/" (minus suffices such as "-leftcam", "-rightcam", and "-bkgd")
# argv[2]: gpu number (0, 1, 2, ...)
```

<br>

To train Total-Recon on your own videos, find the instructions [here](../README.md#training).
