#!/bin/bash
# End-to-end prototype-based pipeline for the contest dataset.
# Does NOT use all_class_predictions.json.
#
# Launch with:
#   setsid nohup bash scripts/run_pipeline_prototype.sh > output/pipeline_prototype.log 2>&1 < /dev/null &
set -u
cd /root/code/NLPrompt
PY=./.venv/bin/python
DATA=/root/datasets/contest
CLIP=/root/weights/ViT-B-32.pt
CLEAN_DIR=output/contest_clean
PROTO_DIR=output/contest_prototype
TRAIN_DIR=output/contest_train

mkdir -p output

echo "[pipeline] $(date) === Step 1: cleaning (damage/degenerate image detection) ==="
${PY} -u clean_contest.py \
    --data-root ${DATA} \
    --output-dir ${CLEAN_DIR} \
    --clip-weights ${CLIP} \
    > output/clean_prototype.log 2>&1
CLEAN_RC=$?
if [ ${CLEAN_RC} -ne 0 ] \
   || [ ! -f ${CLEAN_DIR}/clean_train_manifest.json ] \
   || [ ! -f ${CLEAN_DIR}/clean_test_manifest.json ]; then
    echo "[pipeline] $(date) ERROR: cleaning failed (rc=${CLEAN_RC}). See output/clean_prototype.log"
    exit 1
fi
echo "[pipeline] $(date) cleaning done."

echo "[pipeline] $(date) === Step 2: prototype discovery (per-class k-means denoising) ==="
${PY} -u class_prototype.py \
    --data-root ${DATA} \
    --clean-manifest ${CLEAN_DIR}/clean_train_manifest.json \
    --clip-weights ${CLIP} \
    --output-dir ${PROTO_DIR} \
    --batch-size 256 \
    --num-workers 2 \
    > output/proto.log 2>&1
PROTO_RC=$?
if [ ${PROTO_RC} -ne 0 ] || [ ! -f ${PROTO_DIR}/prototypes.pt ]; then
    echo "[pipeline] $(date) ERROR: prototype discovery failed (rc=${PROTO_RC}). See output/proto.log"
    exit 1
fi
echo "[pipeline] $(date) prototype discovery done."

echo "[pipeline] $(date) === Step 3: training (cosine classifier on cleaned prototypes) ==="
${PY} -u train_prototype.py \
    --prototype-dir ${PROTO_DIR} \
    --output-dir ${TRAIN_DIR} \
    --epochs 200 \
    --batch-size 4096 \
    > output/train_prototype.log 2>&1
TRAIN_RC=$?
if [ ${TRAIN_RC} -ne 0 ] || [ ! -f ${TRAIN_DIR}/best.pt ]; then
    echo "[pipeline] $(date) ERROR: training failed (rc=${TRAIN_RC}). See output/train_prototype.log"
    exit 1
fi
echo "[pipeline] $(date) training done."

echo "[pipeline] $(date) === Step 4: inference ==="
${PY} -u test_prototype.py \
    --checkpoint ${TRAIN_DIR}/best.pt \
    --data-root ${DATA} \
    --clean-test-manifest ${CLEAN_DIR}/clean_test_manifest.json \
    --clip-weights ${CLIP} \
    --output-dir ${TRAIN_DIR} \
    --tta \
    > output/test_prototype.log 2>&1
TEST_RC=$?
if [ ${TEST_RC} -ne 0 ] || [ ! -f ${TRAIN_DIR}/pred_results.zip ]; then
    echo "[pipeline] $(date) ERROR: inference failed (rc=${TEST_RC}). See output/test_prototype.log"
    exit 1
fi
echo "[pipeline] $(date) ALL DONE. Submission: ${TRAIN_DIR}/pred_results.zip"
