src_database_dir=/data2/ndsong/banmo_ds/database/DAVIS
tgt_database_dir=/data2/ndsong/banmo_fgbg/database/DAVIS

seqname=$1

for datatype in `ls -d $src_database_dir/*`; do
    datatype=`basename $datatype`
   
    #if [ $datatype != "Pixels" ]
    #then
    #	echo $datatype
    #    sudo mkdir -p $tgt_database_dir/$datatype/Full-Resolution
    #    sudo cp -r $src_database_dir/$datatype/Full-Resolution/$seqname $tgt_database_dir/$datatype/Full-Resolution/
    #fi

    sudo mkdir -p $tgt_database_dir/$datatype/Full-Resolution
    sudo cp -r $src_database_dir/$datatype/Full-Resolution/$seqname $tgt_database_dir/$datatype/Full-Resolution/

    #sudo cp -r $src_database_dir/$datatype/Full-Resolution/cat-pikachiu-rgbd-bkgd000 $tgt_database_dir/$datatype/Full-Resolution/
    #cp -r $src_database_dir/$datatype/Full-Resolution/cat-pikachiu-rgbd001 $tgt_database_dir/$datatype/Full-Resolution/
    #cp -r $src_database_dir/$datatype/Full-Resolution/cat-pikachiu-rgbd002 $tgt_database_dir/$datatype/Full-Resolution/
    #cp -r $src_database_dir/$datatype/Full-Resolution/cat-pikachiu-rgbd003 $tgt_database_dir/$datatype/Full-Resolution/
done
