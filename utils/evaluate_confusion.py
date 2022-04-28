import numpy as np

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

    count = 0
    conf_matrix = np.zeros((num_classes-1, num_classes-1+num_unknown_classes))
    for (pred_label, true_mapped_label, true_unmapped_label) in zip(pred, true_mapped, true_unmapped):
        # correctly predicted unknown label [U, U]
        if pred_label == unknown_label and pred_label == true_mapped_label:
            continue
        # true label is a known class but predicted unknown class
        if pred_label == unknown_label and true_mapped_label < unknown_label:
            count += 1
            continue
        # predicted as known and true unknown
        if pred_label < unknown_label and true_mapped_label < unknown_label:
            conf_matrix[pred_label][true_mapped_label] += 1
        else:
            mapped_unk_label = unknown_label_to_idx[true_unmapped_label]
            conf_matrix[pred_label][unknown_label + mapped_unk_label] += 1

    # print("Number of known classes predicted as unknown class = ", count) # 515821
    return conf_matrix


    #       Kn 1    Kn2     Kn3     Unk1    Unk2
    # Kn 1
    # Kn 2
    # Kn 3
    # True: [1, 2, 6 -> 40, 5, 6 -> 41], Pred: [1, 2, 3, 1, 2, 1]
    # Step 1: Wherever unk class prediction is right, do nothing
    # Step 2: Wherever unk class prediction is wrong, make this matrix somehow

    # Ensure data is in the right format
    # true_mapped = np.squeeze(true_mapped)
    # pred = np.squeeze(pred)
    # true_unmapped = np.squeeze(true_unmapped)
    # if len(true_mapped.shape) != 1:
    #     raise ValueError('Truth values are stored in a {:d}D array instead of 1D array'. format(len(true.shape)))
    # if len(pred.shape) != 1:
    #     raise ValueError('Prediction values are stored in a {:d}D array instead of 1D array'. format(len(pred.shape)))
    # if true_mapped.dtype not in [np.int32, np.int64]:
    #     raise ValueError('Truth values are {:s} instead of int32 or int64'.format(true.dtype))
    # if pred.dtype not in [np.int32, np.int64]:
    #     raise ValueError('Prediction values are {:s} instead of int32 or int64'.format(pred.dtype))
    # true_mapped = true_mapped.astype(np.int32)
    # pred = pred.astype(np.int32)

    # # Get the label values
    # if label_values is None:
    #     # From data if they are not given
    #     label_values = np.unique(np.hstack((true_mapped, pred)))
    # else:
    #     # Ensure they are good if given
    #     if label_values.dtype not in [np.int32, np.int64]:
    #         raise ValueError('label values are {:s} instead of int32 or int64'.format(label_values.dtype))
    #     if len(np.unique(label_values)) < len(label_values):
    #         raise ValueError('Given labels are not unique')

    # # Sort labels
    # label_values = np.sort(label_values)

    # # Get the number of classes
    # num_classes = len(label_values)

    # #print(num_classes)
    # #print(label_values)
    # #print(np.max(true))
    # #print(np.max(pred))
    # #print(np.max(true * num_classes + pred))

    # # Start confusion computations
    # if label_values[0] == 0 and label_values[-1] == num_classes - 1:

    #     # Vectorized confusion
    #     vec_conf = np.bincount(true_mapped * num_classes + pred)

    #     # Add possible missing values due to classes not being in pred or true
    #     #print(vec_conf.shape)
    #     if vec_conf.shape[0] < num_classes ** 2:
    #         vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')
    #     #print(vec_conf.shape)

    #     # Reshape confusion in a matrix
    #     return vec_conf.reshape((num_classes, num_classes))


    # else:

    #     # Ensure no negative classes
    #     if label_values[0] < 0:
    #         raise ValueError('Unsupported negative classes')

    #     # Get the data in [0,num_classes[
    #     label_map = np.zeros((label_values[-1] + 1,), dtype=np.int32)
    #     for k, v in enumerate(label_values):
    #         label_map[v] = k

    #     pred = label_map[pred]
    #     true_mapped = label_map[true_mapped]

    #     # Vectorized confusion
    #     vec_conf = np.bincount(true_mapped * num_classes + pred)

    #     # Add possible missing values due to classes not being in pred or true
    #     if vec_conf.shape[0] < num_classes ** 2:
    #         vec_conf = np.pad(vec_conf, (0, num_classes ** 2 - vec_conf.shape[0]), 'constant')

    #     # Reshape confusion in a matrix
    #     return vec_conf.reshape((num_classes, num_classes))