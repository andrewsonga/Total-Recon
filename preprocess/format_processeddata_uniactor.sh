# e.g. prefix=human1-mono000, dog1-mono000, cat1-mono000
prefix=$1
gpu=$2

seqname=$prefix
seqname_bkgd=$seqname-bkgd

CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname --norecon_bkgd
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_bkgd --recon_bkgd