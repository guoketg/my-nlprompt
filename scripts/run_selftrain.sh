#!/bin/bash
# Stage 3 self-training: expand clean mask and retrain.
# Run after scripts/run_pipeline_prototype.sh has produced a trained checkpoint.
#
# Launch with:
#   setsid nohup bash scripts/run_selftrain.sh > output/selftrain.log 2>&1 < /dev/null &
set -u
cd /root/code/NLPrompt
PY=./.venv/bin/python
DATA=/root/datasets/contest
CLIP=/root/weights/ViT-B-32.pt
CLEAN_DIR=output/contest_clean
PROTO_DIR=output/contest_prototype
TRAIN_DIR=output/contest_train
SELF_DIR=output/contest_prototype_selftrain
FINAL_DIR=output/contest_train_final

mkdir -p output

echo "[selftrain] $(date) === Step 3a: expand clean mask ==="
${PY} -u self_train_prototype.py \
    --prototype-dir ${PROTO_DIR} \
    --checkpoint ${TRAIN_DIR}/best.pt \
    --output-dir ${SELF_DIR} \
    --conf-thr 0.9 \
    > output/selftrain_expand.log 2>&1
SELF_RC=$?
if [ ${SELF_RC} -ne 0 ] || [ ! -f ${SELF_DIR}/clean_mask.pt ]; then
    echo "[selftrain] $(date) ERROR: self-training expansion failed (rc=${SELF_RC}). See output/selftrain_expand.log"
    exit 1
fi
echo "[selftrain] $(date) clean mask expanded."

echo "[selftrain] $(date) === Step 3b: retrain on expanded clean set ==="
${PY} -u train_prototype.py \
    --prototype-dir ${SELF_DIR} \
    --output-dir ${FINAL_DIR} \
    --epochs 100 \
    --batch-size 4096 \
    --lr 5e-4 \
    > output/selftrain_train.log 2>&1
TRAIN_RC=$?
if [ ${TRAIN_RC} -ne 0 ] || [ ! -f ${FINAL_DIR}/best.pt ]; then
    echo "[selftrain] $(date) ERROR: retraining failed (rc=${TRAIN_RC}). See output/selftrain_train.log"
    exit 1
fi
echo "[selftrain] $(date) retraining done."

echo "[selftrain] $(date) === Step 3c: inference ==="
${PY} -u test_prototype.py \
    --checkpoint ${FINAL_DIR}/best.pt \
    --data-root ${DATA} \
    --clean-test-manifest ${CLEAN_DIR}/clean_test_manifest.json \
    --clip-weights ${CLIP} \
    --output-dir ${FINAL_DIR} \
    --tta \
    > output/selftrain_test.log 2>&1
TEST_RC=$?
if [ ${TEST_RC} -ne 0 ] || [ ! -f ${FINAL_DIR}/pred_results.zip ]; then
    echo "[selftrain] $(date) ERROR: inference failed (rc=${TEST_RC}). See output/selftrain_test.log"
    exit 1
fi
echo "[selftrain] $(date) ALL DONE. Submission: ${FINAL_DIR}/pred_results.zip"
