# e.g. prefix=humandog-stereo000
prefix=$1

# 1a) no depth: fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-nodepth-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-nodepth-animal-leftcam.config

# 1b) no depth: bkgd pretraining
cp configs/$prefix-bkgd-leftcam.config configs/$prefix-nodepth-bkgd-leftcam.config

# 1c) no depth: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-nodepth-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-nodepth-rightcam-jointft.config

# 2a) no deform: fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-nodeform-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-nodeform-animal-leftcam.config

# 2b) no deform: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-nodeform-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-nodeform-rightcam-jointft.config

# 3a) no posenet: fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-noposenet-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-noposenet-animal-leftcam.config

# 3b) no posenet: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-noposenet-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-noposenet-rightcam-jointft.config

# 4a) no rootbody: fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-norootbody-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-norootbody-animal-leftcam.config

# 4b) no rootbody: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-norootbody-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-norootbody-rightcam-jointft.config

# 5a) no cam. opt: bkgd pretraining
cp configs/$prefix-bkgd-leftcam.config configs/$prefix-nocamopt-bkgd-leftcam.config

# 5b) no cam. opt: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-nocamopt-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-nocamopt-rightcam-jointft.config

# 6a) se3: fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-se3-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-se3-animal-leftcam.config

# 6b) se3: joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-se3-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-se3-rightcam-jointft.config

# 7a) no rootbody (se3): fg pretraining
cp configs/$prefix-human-leftcam.config configs/$prefix-norootbody-se3-human-leftcam.config
cp configs/$prefix-animal-leftcam.config configs/$prefix-norootbody-se3-animal-leftcam.config

# 7b) no rootbody (se3): joint finetuning
cp configs/$prefix-leftcam-jointft.config configs/$prefix-norootbody-se3-leftcam-jointft.config
cp configs/$prefix-rightcam-jointft.config configs/$prefix-norootbody-se3-rightcam-jointft.config