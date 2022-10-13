#!/bin/bash

# default value for some args
gpu_id=0
task_set=1

# fetch arguments
while getopts p:e:t:g: flag
do
    case "${flag}" in
        p) prev_train_path=${OPTARG};;
        e) exp_name=${OPTARG};;
        t) task_set=${OPTARG};;
        g) gpu_id=${OPTARG};;
    esac
done

# Support empty experiment name
if [ -z "${exp_name}" ]
  then
    VAL_OUTPUT_RELATIVE="val_preds_TS${task_set}"
    VAL_OUTPUT="test/val_preds_TS${task_set}"
    TRK_OUTPUT="results/predictions/TS${task_set}"
    EVAL_OUTPUT="results/metrics/TS${task_set}"
else
    VAL_OUTPUT_RELATIVE="val_preds_TS${task_set}_${exp_name}"
    VAL_OUTPUT="test/val_preds_TS${task_set}_${exp_name}"
    TRK_OUTPUT="results/predictions/TS${task_set}_${exp_name}"
    EVAL_OUTPUT="results/metrics/TS${task_set}_${exp_name}"
fi

# source activate 4dpls

# Validation
echo "Running Validation"
CUDA_VISIBLE_DEVICES=${gpu_id} python validate_semanticKitti.py -t ${task_set} -p "${prev_train_path}" -s "$VAL_OUTPUT_RELATIVE"
echo "Validation Complete!"

# LOSP
echo "Running LOSP"
cd hu-segmentation/
CUDA_VISIBLE_DEVICES=${gpu_id} python segment_with_ours_write_instances.py -d semantic-kitti -t ${task_set} -s 8 -o "../${VAL_OUTPUT}"
cd ../
echo "LOSP complete!"

# Tracking
echo "Running tracking"
CUDA_VISIBLE_DEVICES=${gpu_id} python 4dpls_tracking_greedy.py -t ${task_set} -p "${VAL_OUTPUT}" -sd "${TRK_OUTPUT}" -dc data/SemanticKitti/semantic-kitti.yaml
echo "Tracking complete!"

# Evaluation
echo "Running evaluation"
mkdir -p "${EVAL_OUTPUT}"
cd utils/
CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate_panoptic.py -d ../data/SemanticKitti -p "../${TRK_OUTPUT}" -dc ../data/SemanticKitti/semantic-kitti.yaml -t ${task_set} -o "../${EVAL_OUTPUT}"
cd ../
echo "Evaluation complete!"