# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

src_dir=$1
tgt_dir=logdir

# directory inside Total-Recon where the pre-optimized models will be stored
# (assumes we've `cd`ed into Total-Recon)
mkdir -p logdir

# copy the pre-optimized models from the source directory to the target directory
cp -r $src_dir/* $tgt_dir/