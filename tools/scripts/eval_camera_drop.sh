#!/usr/bin/env bash
# Evaluate a checkpoint with all cameras working and under the three camera-failure settings.
#   bash tools/scripts/eval_camera_drop.sh CONFIG CHECKPOINT [NUM_GPUS] [extra test.py args]
set -e
CONFIG=$1
CHECKPOINT=$2
GPUS=${3:-1}
WORK_DIR=${WORK_DIR:-work_dirs/$(basename ${CONFIG%.*})}

for DROP in none front1 front3 all; do
    OUT=$WORK_DIR/test_$DROP
    mkdir -p $OUT
    if [ "$GPUS" -gt 1 ]; then
        bash $(dirname $0)/dist_test.sh $CONFIG $CHECKPOINT $GPUS \
            --camera-drop $DROP --work-dir $OUT ${@:4} 2>&1 | tee $OUT/test.log
    else
        python $(dirname $0)/../test.py $CONFIG $CHECKPOINT \
            --camera-drop $DROP --work-dir $OUT ${@:4} 2>&1 | tee $OUT/test.log
    fi
done

echo
printf '%-8s %8s %8s %8s %8s\n' cameras ped div bound mAP
for DROP in none front1 front3 all; do
    python - $WORK_DIR/test_$DROP/test.log $DROP <<'PY'
import ast, sys
line = [l for l in open(sys.argv[1]) if l.startswith("{'ped_crossing'")][-1]
r = ast.literal_eval(line)
print('%-8s %8.1f %8.1f %8.1f %8.1f' % (sys.argv[2], 100 * r['ped_crossing'], 100 * r['divider'],
                                         100 * r['boundary'], 100 * r['mAP']))
PY
done
