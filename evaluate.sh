# bash evaluate.sh $prefix $gpu $targetview "$ablation"

prefix=$1
gpu=$2
targetview=$3       # 'inputview' or 'stereoview'
ablation=$4         # a phrase such as "-nodeform" or empty string "" for full model

# 1) copy left-rightcam registration data to the model directory
# assumes the jointly finetuned model dir. is named as $prefix-leftcam-jointft
seqname=$prefix$ablation-leftcam-jointft

cp raw/$prefix-leftcam/normrefcam2secondcam.npy logdir/$seqname/        # for uniactor sequences
cp raw/$prefix-human-leftcam/normrefcam2secondcam.npy logdir/$seqname/  # for multiactor sequences

echo seqname is $seqname

# 2) extract mesh, bones, and cameras (will be used for mesh rendering and evaluation)
bash extract_fgbg.sh $gpu $seqname

# 3) run evaluation w.r.t. view specified by $targetview
bash scripts/render_nvs_fgbg_$targetview.sh $gpu $seqname

# 4) print metrics
python print_metrics.py --seqname $seqname --view $targetview
