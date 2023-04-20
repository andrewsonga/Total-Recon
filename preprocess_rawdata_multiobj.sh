prefix=$1
isdynamic=y        # y/n
rootdir=raw
tmpdir=tmp

# 1) make tmp if it doesn't exist
mkdir -p $tmpdir

# 2) make copies of raw dataset for per-object data preprocessing
echo making copies of raw dataset for per-object data preprocessing
rm -rf $rootdir/$prefix-human-leftcam
rm -rf $rootdir/$prefix-human-rightcam
rm -rf $rootdir/$prefix-animal-leftcam
rm -rf $rootdir/$prefix-animal-rightcam
cp -r $rootdir/$prefix-leftcam $rootdir/$prefix-human-leftcam
cp -r $rootdir/$prefix-rightcam $rootdir/$prefix-human-rightcam
cp -r $rootdir/$prefix-leftcam $rootdir/$prefix-animal-leftcam
cp -r $rootdir/$prefix-rightcam $rootdir/$prefix-animal-rightcam

# 3) generate preprocessed data for the "human" object
#    this version of the dataset will include masks for the human object, and densepose features extracted for the human object
bash preprocess/preprocess_frames_dualrig.sh $prefix-human y $isdynamic

# 4) generate preprocessed data for the "animal" object
#    this version of the dataset will include masks for the animal object, and densepose features extracted for the animal (quadruped) object
bash preprocess/preprocess_frames_dualrig.sh $prefix-animal n $isdynamic

# 5) generate preprocessed data for the "background" object
#    by copying the contents of that of the "human" object
#    (assumes there only one sequence i.e. 000 with given $prefix)
bash cp_database.sh $prefix-human000-leftcam $prefix-bkgd000-leftcam
bash cp_database.sh $prefix-human000-rightcam $prefix-bkgd000-rightcam

# [optional] remove generated copies of raw dataset in step 1
#rm -rf $rootdir/$prefix-human-leftcam
#rm -rf $rootdir/$prefix-human-rightcam
#rm -rf $rootdir/$prefix-animal-leftcam
#rm -rf $rootdir/$prefix-animal-rightcam