seqname=$1
drivenmodel=$2
retargetview=$3
#python preprocess/img2lines.py --seqname $seqname
bash scripts/template-retarget.sh 0,1,2,3 $seqname 10001 "" "no" $drivenmodel
bash scripts/render_nvs.sh 0 $seqname logdir/driver-$seqname-e30-b128/params_latest.pth $retargetview $retargetview
