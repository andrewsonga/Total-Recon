src_dir=$1
tgt_dir=database

# directory inside Total-Recon where the preprocessed data will be stored
# (assumes we've `cd`ed into Total-Recon)
mkdir -p $tgt_dir/DAVIS

# copy the preprocessed data from the source directory to the target directory
for datatype in `ls -d $src_dir/DAVIS/*`; do
    datatype=`basename $datatype`
    
    if [ $datatype != "Pixels" ]
    then
        echo copying $datatype
        cp -r $src_dir/DAVIS/$datatype/Full-Resolution/* $tgt_dir/DAVIS/$datatype/Full-Resolution/    
    fi
done