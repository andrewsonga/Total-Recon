prefix=$1
ishuman=$2          # y/n
isdynamic=y         # y/n

# generate preprocessed data
bash preprocess/preprocess_frames_dualrig.sh $prefix $ishuman $isdynamic