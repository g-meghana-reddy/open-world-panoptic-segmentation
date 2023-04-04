#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on SemanticKitti dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import glob
import argparse
import signal
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from datasets.SemanticKitti import *
from datasets.Kitti360 import *
from models.architectures import KPFCNN
from utils.config import Config
from utils.evaluate_confusion import confusion_openset
from utils.metrics import fast_confusion
from utils.trainer import ModelTrainer
from validate_kitti360 import Kitti360Config
from validate_semanticKitti import SemanticKittiConfig

import sklearn
from tqdm import tqdm
import pdb


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Parse args
#       \******************/
#


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set",
                        help="Task Set ID", type=int, default=2)
    parser.add_argument("-p", "--prev_train_path",
                        help="Directory to load checkpoint", default=None)
    parser.add_argument("-i", "--chkp_idx",
                        help="Index of checkpoint",  default=None)
    parser.add_argument('-sk', "--semantic_kitti",
                        dest='semantic_kitti', action="store_true", default=False)
    parser.add_argument('-k360', "--kitti360", dest='kitti360',
                        action="store_true", default=False)
    parser.add_argument(
        "-s", "--seq", help="Sequence number", type=int, default=8)
    args = parser.parse_args()
    return args


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################
    args = parse_args()

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    # chkp_idx = None
    previous_training_path = args.prev_train_path
    chkp_idx = args.chkp_idx

    # Find all snapshot in the chosen training folder
    chkp_path = os.path.join('results', 'checkpoints',
                             previous_training_path, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(
        'results', 'checkpoints', previous_training_path, 'checkpoints', chosen_chkp)

    ##############
    # Prepare Data
    ##############
    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    if args.semantic_kitti:
        config = SemanticKittiConfig()
    if args.kitti360:
        config = Kitti360Config()

    if previous_training_path:
        config.load(os.path.join(
            'results', 'checkpoints', previous_training_path))
        config.saving_path = None
    config.pre_train = False  # True
    config.free_dim = 4
    config.n_frames = 1  # 4
    config.n_test_frames = 1
    config.stride = 1
    config.sampling = 'importance'
    config.decay_sampling = 'None'

    if args.semantic_kitti:
        config.validation_size = 4071
        dataset_name = 'semantic_kitti'

    if args.kitti360:
        data_dir = 'data/Kitti360'
        seq_dir = os.path.join(data_dir, 'data_3d_raw_labels',
                               '2013_05_28_drive_{:04d}_sync'.format(args.seq), 'labels')
        config.epoch_steps = len(glob.glob(seq_dir + '/*.label'))
        config.validation_size = config.epoch_steps
        config.sequence = args.seq
        dataset_name = 'kitti360'

    config.task_set = args.task_set

    if config.task_set in [0, 1, 2]:
        return_unknowns = True
    else:
        return_unknowns = False

    if args.semantic_kitti:
        # Initialize datasets
        test_dataset = SemanticKittiDataset(config, set='validation',
                                            balance_classes=False,
                                            return_unknowns=return_unknowns,
                                            seqential_batch=True)

    if args.kitti360:
        # Initialize datasets
        test_dataset = Kitti360Dataset(config, split='validation',
                                       balance_classes=False,
                                       return_unknowns=return_unknowns,
                                       seqential_batch=True)

    # Initialize samplers
    test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()

    checkpoint = torch.load(chosen_chkp)

    net = KPFCNN(config, test_dataset.label_values,
                 test_dataset.ignored_labels)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Define a trainer class
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart forward pass')
    print('**************')

    softmax = torch.nn.Softmax(1)
    predictions = []
    true_mapped_labels = []
    true_unmapped_labels = []

    val_label_values = test_loader.dataset.label_values
    num_classes = len(val_label_values)
    for batch in tqdm(test_loader):
        if torch.cuda.device_count() >= 1:
            net.to(torch.cuda.current_device())
            batch.to(torch.cuda.current_device())

        with torch.no_grad():
            outputs, centers_output, var_output, embedding = net(batch, config)
            probs = softmax(outputs).cpu().detach().numpy()
            preds = val_label_values[np.argmax(probs, axis=1)]
            preds = torch.from_numpy(preds)
            preds.to(outputs.device)
        # Get probs and labels
        stk_probs = softmax(outputs).cpu().detach().numpy()
        lengths = batch.lengths[0].cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels
        if config.task_set in [0, 1, 2]:
            unknown_labels_list = batch.val_unk_labels_list
            unknown_label_values = list(
                test_dataset.unknown_label_to_names.keys())
            num_unknown_classes = len(unknown_label_values)

        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            frame_labels = labels_list[b_i]
            if config.task_set in [0, 1, 2]:
                frame_unknown_labels = unknown_labels_list[b_i]

            # Project predictions on the frame points
            proj_probs = probs[proj_inds]

            # Safe check if only one point:
            if proj_probs.ndim < 2:
                proj_probs = np.expand_dims(proj_probs, 0)

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_label_values):
                if label_value in test_loader.dataset.ignored_labels:
                    proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_label_values[np.argmax(proj_probs, axis=1)]

            predictions += [preds]
            true_mapped_labels += [frame_labels[proj_mask]]
            if config.task_set in [0, 1, 2]:
                true_unmapped_labels += [frame_unknown_labels[proj_mask]]

    print('\nCreate confusion matrix')
    print('**************')

    if args.task_set == 1:
        k = 6
    elif args.task_set == 2:
        k = 10
    else:
        k = 19

    if args.task_set == -1:
        conf_matrix_1 = sklearn.metrics.confusion_matrix(
            np.concatenate(true_mapped_labels),
            np.concatenate(predictions),
            labels=val_label_values)

    else:
        conf_matrix_1 = sklearn.metrics.confusion_matrix(
            np.concatenate(true_mapped_labels),
            np.concatenate(predictions),
            labels=val_label_values)

        conf_matrix_2 = confusion_openset(
            np.concatenate(true_mapped_labels),
            np.concatenate(predictions),
            np.concatenate(true_unmapped_labels),
            val_label_values,
            unknown_label_values)

    # Remove ignored labels from confusions
    conf_matrix_1 = np.delete(conf_matrix_1, 0, axis=0)
    conf_matrix_1 = np.delete(conf_matrix_1, 0, axis=1)

    # Balance with real validation proportions
    if args.task_set == -1:
        conf_matrix_1 = conf_matrix_1.T
        conf_matrix_1 = conf_matrix_1.astype(np.float64)
        conf_matrix_1 /= np.expand_dims(
            (np.sum(conf_matrix_1, axis=1) + 1e-6), 0)
        y_labels = np.array(test_dataset.label_names)[1:]
        x_labels = y_labels

        plt.figure(figsize=(30, 10))
        sns.heatmap(conf_matrix_1, xticklabels=x_labels,
                    yticklabels=y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig(
            'results/updated_confusion_matrix_ts{}_{}_balanced.png'.format(args.task_set, dataset_name))

    else:
        # Unknown to known confusion
        conf_matrix_1 = conf_matrix_1.T
        unk_to_known_conf = np.zeros(
            (num_classes - 1, num_classes - 2 + num_unknown_classes - 1))

        unk_to_known_conf[:, num_classes - 2:] = conf_matrix_2
        unk_to_known_conf[:, :num_classes -
                          2] = conf_matrix_1[:, :num_classes-2]
        unk_to_known_conf /= np.expand_dims(
            np.sum(unk_to_known_conf, axis=0) + 1e-6, 0)

        unk_to_known_y_labels = np.array(test_dataset.label_names)[1:-1]
        unk_to_known_x_labels = np.concatenate(
            [unk_to_known_y_labels, test_dataset.unknown_label_names[:-1]])
        unk_to_known_y_labels = np.array(test_dataset.label_names)[1:]

        plt.figure(figsize=(20, 10))
        sns.heatmap(unk_to_known_conf, xticklabels=unk_to_known_x_labels,
                    yticklabels=unk_to_known_y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig(
            'results/updated_extended_confusion_matrix_ts{}_{}_balanced.png'.format(args.task_set, dataset_name))

        # Known to Unknown confusion
        conf_matrix = conf_matrix_1.astype(np.float64)
        conf_matrix /= np.expand_dims(np.sum(conf_matrix, axis=0) + 1e-6, 0)
        y_labels = np.array(test_dataset.label_names)[1:]
        x_labels = y_labels

        plt.figure(figsize=(20, 10))
        sns.heatmap(conf_matrix, xticklabels=x_labels,
                    yticklabels=y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig(
            'results/updated_normal_confusion_matrix_ts{}_{}_balanced.png'.format(args.task_set, dataset_name))
