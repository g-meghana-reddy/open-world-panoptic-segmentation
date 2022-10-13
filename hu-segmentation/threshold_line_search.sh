for i in {0,0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.5}; do
    cd /project_data/ramanan/achakrav/4D-PLS/utils/
    pred_dir="../results/predictions/4dpls_huseg_agnostic_thresh_$i"
    cmd="python3 segment_with_ours_class_agnostic.py -t -1 -o $pred_dir --threshold $i"
    # echo $cmd
    # $cmd &
    output_dir="../results/metrics/4dpls_huseg_agnostic_thresh_$i"
    mkdir -p $output_dir
    cmd="python3 evaluate_panoptic.py -d ../data/SemanticKitti -p $pred_dir -dc ../data/SemanticKitti/semantic-kitti-orig.yaml -t -1 -o $output_dir"
    # echo $cmd
    $cmd &
cd ../hu-segmentation
done