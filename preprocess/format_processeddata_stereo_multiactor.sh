# e.g. prefix=humandog-stereo000, humancat-stereo000
prefix=$1
gpu=$2

# used for pretraining
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-human-leftcam --norecon_bkgd         
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-animal-leftcam --norecon_bkgd
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-bkgd-leftcam --recon_bkgd

# used for joint finetuning
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-human-uncropped-leftcam --recon_bkgd         
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-animal-uncropped-leftcam --recon_bkgd

# the "--norecon_bkgd" flag crops the gt observations around the detected foreground actor
# the "--recon_bkgd" flag uses the full-scale gt observations, capturing both the foreground actor and ambient background