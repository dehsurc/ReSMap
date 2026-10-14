#!/usr/bin/env bash
# bash tools/scripts/dist_test.sh CONFIG CHECKPOINT NUM_GPUS [extra test.py args]
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29510}

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/../test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
