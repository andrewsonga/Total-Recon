# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

prefix=$1
ishuman=$2          # y/n
isdynamic=y         # y/n
tmpdir=tmp

# 1) make tmp if it doesn't exist
mkdir -p $tmpdir

# 2) generate preprocessed data
bash preprocess/preprocess_frames_dualrig.sh $prefix $ishuman $isdynamic