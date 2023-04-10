# human, w/ mask loss
seqname=human-dualrig002-properlyscaledcams-leftcam-ds-finetune-fgbkgd
loadname_fg=human-dualrig002-leftcam-e120-b256-ft2
loadname_bkgd=human-dualrig-fgbg002-properlyscaledcams-leftcam-test-e120-b256-ft2
./train_human_fgbg.sh $seqname $loadname_fg $loadname_bkgd 0.5 0.1 1.0 0.1
./extract_fgbg.sh 0 $seqname 0 0

# human, w/o mask loss
seqname=human-dualrig002-properlyscaledcams-leftcam-ds-silwt0-finetune-fgbkgd
loadname_fg=human-dualrig002-leftcam-e120-b256-ft2
loadname_bkgd=human-dualrig-fgbg002-properlyscaledcams-leftcam-test-e120-b256-ft2
./train_human_fgbg.sh $seqname $loadname_fg $loadname_bkgd 0.5 0 1.0 0.1
./extract_fgbg.sh 0 $seqname 0 0

# catamelie, w/ mask loss
seqname=catamelie-dualrig002-properlyscaledcams-leftcam-ds-finetune-fgbkgd
loadname_fg=catamelie-dualrig002-leftcam-e120-b256-ft2
loadname_bkgd=catamelie-dualrig002-fgbg-properlyscaledcams-leftcam-e120-b256-ft2
./train_animal_fgbg.sh $seqname $loadname_fg $loadname_bkgd 0.5 0.1 1.0 0.1
./extract_fgbg.sh 0 $seqname 0 0

# catamelie, w/o mask loss
seqname=catamelie-dualrig002-properlyscaledcams-leftcam-ds-silwt0-finetune-fgbkgd
loadname_fg=catamelie-dualrig002-leftcam-e120-b256-ft2
loadname_bkgd=catamelie-dualrig002-fgbg-properlyscaledcams-leftcam-e120-b256-ft2
./train_animal_fgbg.sh $seqname $loadname_fg $loadname_bkgd 0.5 0 1.0 0.1
./extract_fgbg.sh 0 $seqname 0 0