#!/usr/bin/env bash
# Three-stage ReSMap training (6 + 6 + 12 epochs): bash tools/scripts/train_resmap.sh NUM_GPUS
set -e
GPUS=${1:?usage: bash tools/scripts/train_resmap.sh NUM_GPUS}

bash tools/scripts/dist_train.sh configs/resmap_stage1.py $GPUS
bash tools/scripts/dist_train.sh configs/resmap_stage2.py $GPUS
# stage 3 starts from the best stage-2 checkpoint
STAGE2_BEST=$(ls work_dirs/resmap_stage2/best_mAP_iter_*.pth | head -1)
bash tools/scripts/dist_train.sh configs/resmap_stage3.py $GPUS --cfg-options load_from=$STAGE2_BEST
