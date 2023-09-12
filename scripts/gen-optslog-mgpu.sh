# This code is built upon the BANMo repository: https://github.com/facebookresearch/banmo.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# ==========================================================================================
#
# Carnegie Mellon University’s modifications are Copyright (c) 2023, Carnegie Mellon University. All rights reserved.
# Carnegie Mellon University’s modifications are licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
# To view a copy of the license, visit LICENSE.md.
#
# ==========================================================================================

export MKL_SERVICE_FORCE_INTEL=1
dev=$1
ngpu=`echo $dev |  awk -F '[\t,]' '{print NF-1}'`
ngpu=$(($ngpu + 1 ))
echo "using "$ngpu "gpus"

optslog=$2
logname=$3
seqname=$4
address=$5
add_args=${*: 5:$#-1}

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch\
                    --master_port $address \
                    --nproc_per_node=$ngpu gen_optslog.py \
                    --ngpu $ngpu \
                    --optslog $optslog \
                    --seqname $seqname \
                    --logname $logname \
                    $add_args
