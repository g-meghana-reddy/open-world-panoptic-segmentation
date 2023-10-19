# Lidar Panoptic Segmentation in an Open World

## Step 1: Open-set recognition using 4D-PLS

### Training
Usage:
```
python scripts/train_TS1.sh
```

### Evaluation
Usage:
```
sh scripts/test.sh -t 1 -p 4DPLS_TS1 -e "EXPERIMENT_NAME"
```
One can also optionally specify GPU ID using the `-g` flag.

### Extended Confusion Matrix
To generate confusion and extended confusion matrix, run the following command:
```
python evaluate_conf_matrix.py -t 0 -p ../project_data/ramanan/achakrav/4D-PLS/results/4DPLS_TS0
```

### Visualization

First, go to the `semantic-kitti-api` repo.

Then, generate per-frame visualizations using the following command:
```
python visualize.py -d ../data/SemanticKitti/ -t 1 -c ../data/SemanticKitti/semantic-kitti.yaml -s 08 -p ../results/predictions/TS1 -di --visu 1 -sd ../results/visualizations/trk_valid_vis/TS1
```

Then, go to the directory where png files are saved.
```
cd ../results/visualizations/trk_valid_vis/TS1/1_pred_TS1
```

Finally, run the following command:
```
ffmpeg -r 10 -i %d.png -vf scale=1620:1080 -vframes 500 ../1_pred_TS1.mp4
```

NOTE: For ffmpeg commands, refer to [this link](https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/).

## Step 5: Evaluation on KITTI-360

### Metric evaluation
Run evaluation on a given task set using the following command:
```
sh scripts/test_kitti360.sh -t 1 -p 4DPLS_TS1 -e "EXPERIMENT_NAME"
```

If you wish to evaluate on multiple sequences (currently fixed to sequence 2), change L142 to the appropriate sequence.

### Visualization
First, go to the `semantic-kitti-api` repo.

Then, generate per-frame visualizations using the following command:
```
python visualize_kitti360.py -d ../data/Kitti360/ -t 1 -c ../data/Kitti360/kitti-360.yaml -s 02 -p ../results/predictions/Kitti360/TS1 -di --visu 1 -sd ../results/visualizations/trk_valid_vis/Kitti360/TS1
```

Then, go to the directory where png files are saved.
```
cd ../results/visualizations/trk_valid_vis/Kitti360/TS1/1_pred_TS1/
```

Finally, run the following command:
```
ffmpeg -r 10 -i %d.png -vf scale=1620:1080 -vframes 500 ../1_pred_TS1.mp4
```
