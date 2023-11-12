# e.g. prefix=humandog-mono000, humancat-mono000
prefix=$1
gpu=$2

seqname=$prefix
seqname_human=$seqname-human
seqname_human_uncropped=$seqname_human-uncropped
seqname_animal=$seqname-animal
seqname_animal_uncropped=$seqname_animal-uncropped
seqname_bkgd=$seqname-bkgd

# used for pretraining
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_human --norecon_bkgd         
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_animal --norecon_bkgd
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_bkgd --recon_bkgd

# used for joint finetuning
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_human_uncropped --recon_bkgd         
CUDA_VISIBLE_DEVICES=$gpu python preprocess/img2lines.py --seqname $seqname_animal_uncropped --recon_bkgd

# the "--norecon_bkgd" flag crops the gt observations around the detected foreground actor
# the "--recon_bkgd" flag uses the full-scale gt observations, capturing both the foreground actor and ambient background