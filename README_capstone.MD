# Continual LiDAR Panoptic Segmentation

## Step 1: Open-set recognition using 4D-PLS

### Training
Usage:
```
python scripts/train_TS1.sh
```

### Inference
Usage:
```
python validate_semanticKitti.py -t 1 -p 4DPLS_TS1/ -s val_preds_TS1
```
**Note:** Do not add `/` in the saving path flag. Concretely, use `-s val_preds_TS1` and not `-s val_preds_TS1/`.

## Step 2: Instance mining 
Need to run following commands in `hu-segmentation` repo. 
Usage:
```
python segment_with_ours_write_instances.py -d semantic-kitti -t 1 -s 8
```
Set `task_set` on L114.

## Step 3: Tracking
In order to assign temporally consistent instance IDs, run the following command:
```
python 4dpls_tracking_greedy.py -t 1 -p test/val_preds_TS1 -sd results/validation/TS1/ -dc data/SemanticKitti/semantic-kitti.yaml
```

This generates `.label` files.

## Step 4: Evaluation on SemanticKITTI

### Metrics

If tracking has not been performed, generate `.label` files from semantic and instance predictions using the following command:
```
cd utils/
python create_binary_prediction.py
```
Change `task_set` on L14 and `segmented_unknown` on L23. Note that this command should be run for Task set 2 and when tracking is not performed.

Then, generate panoptic segmentation metrics using the following command:
```
python evaluate_panoptic.py -d ../data/SemanticKitti -p ../results/validation/TS1 -dc ../data/SemanticKitti/semantic-kitti.yaml -t 1 -o ../PQ_TS1
```

### Extended Confusion Matrix
To generate confusion and extended confusion matrix, run the following command:
```
python evaluate_conf_matrix.py -t 0 -p ../project_data/ramanan/achakrav/4D-PLS/results/4DPLS_TS0
```

### Visualization

First, go to the `semantic-kitti-api` repo.

Then, generate per-frame visualizations using the following command:
```
python visualize.py -d ../data/SemanticKitti/ -t 1 -c ../data/SemanticKitti/semantic-kitti.yaml -s 08 -p ../results/validation/TS1 -di --visu 1 -sd ../trk_valid_vis/TS1
```

Then, go to the directory where png files are saved.
```
cd ../trk_valid_vis/TS1/1_pred_TS1
```

Finally, run the following command:
```
ffmpeg -r 10 -i %d.png -vf scale=1620:1080 -vframes 500 ../1_pred_TS1.mp4
```

NOTE: For ffmpeg commands, refer to [this link](https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/).

## Step 5: Evaluation on KITTI-360

### Inference
To run inference on sequence number 2, run the following command:
```
python validate_kitti360.py -t 1 -p 4DPLS_TS1 -sd val_preds_TS1_kitti360 -s 2
```

### Instance Mining
Then, run hu-segmentation with the following command:
```
python segment_with_ours_write_instances.py -s 2 -t 1 -d kitti-360
```

### Tracking
To perform tracking on KITTI-360 using the following command:
```
python 4dpls_tracking_greedy_kitti360.py -t 1 -p test/val_preds_TS1_kitti360 -sd results/validation/Kitti360/TS1 -dc data/Kitti360/kitti-360.yaml
```

### Metric evaluation
Run evaluation on a given task set using the following command:
```
python evaluate_panoptic_kitti360.py -d ../data/Kitti360/ -p ../results/validation/Kitti360/TS1 -dc ../data/Kitti360/kitti-360.yaml -t 1 -o ../PQ_TS1_kitti360
```
If you wish to evaluate on multiple sequences (currently fixed to sequence 2), change L142 to the appropriate sequence.

### Visualization
First, go to the `semantic-kitti-api` repo.

Then, generate per-frame visualizations using the following command:
```
python visualize_kitti360.py -d ../data/Kitti360/ -t 1 -c ../data/Kitti360/kitti-360.yaml -s 02 -p ../results/validation/Kitti360/TS1 -di --visu 1 -sd ../trk_valid_vis/Kitti360/TS1
```

Then, go to the directory where png files are saved.
```
cd ../trk_valid_vis/Kitti360/TS1/1_pred_TS1/
```

Finally, run the following command:
```
ffmpeg -r 10 -i %d.png -vf scale=1620:1080 -vframes 500 ../1_pred_TS1.mp4
```
