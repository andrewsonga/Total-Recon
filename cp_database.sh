# Copyright (c) 2023, Carnegie Mellon University. All rights reserved.

src_seqname=$1
tgt_seqname=$2
src_database_dir=database/DAVIS

for datatype in `ls -d $src_database_dir/*`; do
    datatype=`basename $datatype`
    
    if [ $datatype != "Pixels" ]
    then
        echo $datatype
        echo copy_src_dir $src_database_dir/$datatype/Full-Resolution/$src_seqname
        echo copy_tgt_dir $src_database_dir/$datatype/Full-Resolution/$tgt_seqname
        cp -r $src_database_dir/$datatype/Full-Resolution/$src_seqname $src_database_dir/$datatype/Full-Resolution/$tgt_seqname    
    fi

done