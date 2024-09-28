# Lidar Panoptic Segmentation in an Open World

![Gif](https://github.com/user-attachments/assets/2c3c2ab0-9bf1-4802-9e38-9bfe9c02c83a)

This repo provides supporting code for the paper: 

Lidar Panoptic Segmentation in an Open World. IJCV 2024. Anirudh S Chakravarthy, Meghana Reddy Ganesina, Peiyun Hu, Laura Leal-Taixé, Shu Kong, Deva Ramanan, and Aljosa Osep.

* [IJCV Paper](https://link.springer.com/article/10.1007/s11263-024-02166-9)
* [Arxiv](https://arxiv.org/pdf/2409.14273)
* [Code](https://github.com/g-meghana-reddy/open-world-panoptic-segmentation)


This code builds on the PyTorch implementation of [4D-PLS]([url](https://github.com/MehmetAygun/4D-PLS)). Below, we provide instructions to train and evaluate our method, OWL.
![fig](https://github.com/user-attachments/assets/03319205-b9b0-4f4c-86cf-3a00cb5ae211)


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

## Step 2: Evaluation on KITTI-360

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


## Citation 

If you find our code useful, please consider citing our paper:

```
@article{chakravarthy2024lidar,
  title={Lidar Panoptic Segmentation in an Open World},
  author={Chakravarthy, Anirudh S and Ganesina, Meghana Reddy and Hu, Peiyun and Leal-Taix{\'e}, Laura and Kong, Shu and Ramanan, Deva and Osep, Aljosa},
  journal={International Journal of Computer Vision},
  pages={1--22},
  year={2024},
  publisher={Springer}
}
```
