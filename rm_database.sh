src_seqname=$1
src_database_dir=database/DAVIS

for datatype in `ls -d $src_database_dir/*`; do
    datatype=`basename $datatype`    

    echo $datatype
    rm -rf $src_database_dir/$datatype/Full-Resolution/$src_seqname

done