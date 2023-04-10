seqname=$1
dep_wt=$2
sil_wt=$3
flow_wt=$4
ent_wt=$5
add_args=${*: 6:$#-1}
bash scripts/template-fgbg.sh 0,1,2,3 $seqname 10001 "no" "no" $dep_wt $sil_wt $flow_wt $ent_wt $add_args

# $dep_wt $sil_wt $flow_wt $ent_wt: 0.5 0.1 1.0 0.1 by default

# run script as follows:
# ./train_animal_fgbg.sh $seqname --loadname_objs $loadname_fg --loadname_objs $loadname_bkgd 0.5 0 1.0 0.01