# e.g. prefix=human1-stereo000, dog1-stereo000, cat1-stereo000
prefix=$1
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-leftcam --norecon_bkgd
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $prefix-bkgd-leftcam --recon_bkgd