src_dir=$1
tgt_dir=raw

# directory inside Total-Recon where the raw data will be stored
# (assumes we've `cd`ed into Total-Recon)
mkdir -p $tgt_dir

# copy the raw data from the source directory to the target directory
cp -r $src_dir/* $tgt_dir/