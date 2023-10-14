import numpy as np
from tqdm import tqdm
import pdb


def confusion_openset(true_mapped, pred, true_unmapped, label_values=None, unknown_label_values=None):
    """
    Fast confusion matrix (100x faster than Scikit learn). But only works if labels are la
    :param true:
    :param false:
    :param num_classes:
    :return:
    """

    if len(true_mapped.shape) != 1:
        raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true_mapped.shape)))
    if len(pred.shape) != 1:
        raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
    if true_mapped.dtype not in [np.int32, np.int64]:
        raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true_mapped.dtype))
    if pred.dtype not in [np.int32, np.int64]:
        raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
    true_mapped = true_mapped.astype(np.int32)
    pred = pred.astype(np.int32)
    true_unmapped = true_unmapped.astype(np.int32)
    
    # Get the label values
    if label_values is None:
        # From data if they are not given
        label_values = np.unique(np.hstack((true_mapped, pred)))
    else:
        # Ensure they are good if given
        if label_values.dtype not in [np.int32, np.int64]:
            raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
        if len(np.unique(label_values)) < len(label_values):
            raise ValueError('Given labels are not unique')

    # Sort labels
    label_values = np.sort(label_values)

    # Get the number of classes
    num_classes = len(label_values)

    unknown_label = label_values.max()

    # Get the unknown label values
    num_unknown_classes = len(unknown_label_values)
    unknown_label_to_idx = {l: i for i, l in enumerate(unknown_label_values)}
    
    conf_matrix = np.zeros((num_classes, num_unknown_classes-1))
    for (pred_label, true_mapped_label, true_unmapped_label) in tqdm(zip(pred, true_mapped, true_unmapped)):
        # Two cases we are interested:
        # 1. correctly predicted unknown label [U, U]: TP
        # 2. predicted as known class: FN
        if true_mapped_label == unknown_label:
            mapped_unk_label = unknown_label_to_idx[true_unmapped_label]
            conf_matrix[pred_label][mapped_unk_label] += 1

    conf_matrix = np.delete(conf_matrix, 0, axis=0)
    return conf_matrix


    