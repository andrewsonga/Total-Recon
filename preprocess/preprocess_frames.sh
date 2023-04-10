# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#
# bash preprocess/preprocess.sh Sultan .MOV no 10 
#                             folder, file ext, human or not, fps
# file ext can be {.MOV, .mp4, .txt}

rootdir=raw/
tmpdir=tmp/
prefix=$1
filedir=$rootdir/$prefix
maskoutdir=$rootdir/output
finaloutdir=database/DAVIS/
ishuman=$2 # y/n

## rename to upper case
#if [ "$suffix" = ".MOV" ]; then
#  cd $filedir
#  for file in ./*; do mv -- "$file" "${file^^}"; done
#  cd -
#fi

# create required dirs
#mkdir ./tmp
#mkdir -p database/DAVIS/
mkdir -p raw/output

#for infile in `ls -v $filedir/*$suffix`; do
counter=0
for infile in `ls -d $filedir/*`; do              # filedir = raw/$prefix (e.g. raw/andrew)
  echo $infile
  seqname=$prefix$(printf "%03d" $counter)        # prefix = sequence name
  echo $seqname

  # segmentation
  todir=$tmpdir/$seqname                          # tmpdir = tmp/
  rm -rf $todir                                   # todir = tmp/$seqname
  mkdir $todir
  mkdir $todir/images/
  mkdir $todir/masks/
  
  # copies the provided frames in /raw/$seqname/.../images/ into tmp/$seqname_num/images
  cp $infile/images/* $todir/images

  rm -rf $finaloutdir/JPEGImages/Full-Resolution/$seqname  
  rm -rf $finaloutdir/Annotations/Full-Resolution/$seqname 
  rm -rf $finaloutdir/Densepose/Full-Resolution/$seqname   
  mkdir -p $finaloutdir/JPEGImages/Full-Resolution/$seqname
  mkdir -p $finaloutdir/Annotations/Full-Resolution/$seqname
  mkdir -p $finaloutdir/Densepose/Full-Resolution/$seqname
  python preprocess/mask.py $seqname $ishuman

  # copies depthMaps and confidenceMaps into database/DAVIS/ (assume the names aren't always numbered from 0 )
  src_depth_dir=$infile/depths
  src_conf_dir=$infile/confs
  tgt_depth_dir=$finaloutdir/DepthMaps/Full-Resolution/$seqname
  tgt_conf_dir=$finaloutdir/ConfidenceMaps/Full-Resolution/$seqname

  mkdir -p $tgt_depth_dir
  mkdir -p $tgt_conf_dir

  depth_counter=0
  for depthmap in `ls $src_depth_dir/*.depth`; do
      echo $tgt_depth_dir/$(printf "%05d.depth" $depth_counter)
      cp $depthmap $tgt_depth_dir/$(printf "%05d.depth" $depth_counter)
      depth_counter=$((depth_counter+1))
  done

  conf_counter=0
  for confmap in `ls $src_conf_dir/*.conf`; do
      echo $tgt_conf_dir/$(printf "%05d.conf" $conf_counter)
      cp $confmap $tgt_conf_dir/$(printf "%05d.conf" $conf_counter)
      conf_counter=$((conf_counter+1))
  done

  # densepose
  python preprocess/compute_dp.py $seqname $ishuman

  # flow
  cd third_party/vcnplus
  bash compute_flow.sh $seqname
  cd -

  ## Optionally run SfM for initial root pose
  #bash preprocess/colmap_to_data.sh $seqname $ishuman

  ## save to zips
  #cd database/DAVIS/
  #rm -i  $rootdir/$seqname.zip
  #zip $rootdir/$seqname.zip -r  */Full-Resolution/$seqname/
  #cd -

  counter=$((counter+1))
done

# write config file
python preprocess/write_config.py ${seqname::-3} $ishuman