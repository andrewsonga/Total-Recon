prefix=$1
ishuman=$2          # y/n
isdynamic=$3        # y/n

# generate preprocessed data
bash preprocess/preprocess_frames_dualrig.sh $prefix $ishuman $isdynamic