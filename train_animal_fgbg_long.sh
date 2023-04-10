seqname=$1
loadname_fg=$2
loadname_bkgd=$3
dep_wt=$4
sil_wt=$5
flow_wt=$6
ent_wt=$7
bash scripts/template-fgbg-long.sh 0,1,2,3 $seqname $loadname_fg $loadname_bkgd 10001 "no" "no" $dep_wt $sil_wt $flow_wt $ent_wt

#$dep_wt $sil_wt $flow_wt $ent_wt: 0.5 0.1 1.0 0.1 by default