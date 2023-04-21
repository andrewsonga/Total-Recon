# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

src_dir=$1
tgt_dir=$2

# directory inside Total-Recon where the preprocessed data will be stored
# (assumes we've `cd`ed into Total-Recon)
mkdir -p $tgt_dir/DAVIS

# copy the preprocessed data from the source directory to the target directory
for datatype in `ls -d $src_dir/DAVIS/*`; do
    datatype=`basename $datatype`
    
    if [ $datatype != "Pixels" ]
    then
        echo copying $datatype
        mkdir -p $tgt_dir/DAVIS/$datatype/Full-Resolution
        cp -r $src_dir/DAVIS/$datatype/Full-Resolution/* $tgt_dir/DAVIS/$datatype/Full-Resolution/    
    fi
done