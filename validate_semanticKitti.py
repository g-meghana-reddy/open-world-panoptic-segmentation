# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import argparse
import signal
import os

# Dataset
from datasets.SemanticKitti import *
from models.architectures import KPFCNN
from utils.config import Config
from utils.trainer import ModelTrainer
from utils.tester import ModelTester

import pdb

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class SemanticKittiConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'SemanticKitti'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Task set to be selected on the dataset
    task_set = 2

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Radius of the input sphere
    in_radius = 6.0
    val_radius = 51.0
    n_frames = 1  # 4
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 8
    val_batch_num = 1

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 256
    in_features_dim = 2
    free_dim = 3

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    # Multiplier for learning rate applied to the deformations
    deform_lr_factor = 0.1
    # Distance of repulsion for deformed kernel points
    repulse_extent = 1.2

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1000

    # Learning rate management
    learning_rate = 1e-4
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 200) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of steps per epochs
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 200

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    # class proportion for R=10.0 and dl=0.08 (first is unlabeled)
    # 19.1 48.9 0.5  1.1  5.6  3.6  0.7  0.6  0.9 193.2 17.7 127.4 6.7 132.3 68.4 283.8 7.0 78.5 3.3 0.8
    #
    #

    # sqrt(Inverse of proportion * 100)
    # class_w = [1.430, 14.142, 9.535, 4.226, 5.270, 11.952, 12.910, 10.541, 0.719,
    #            2.377, 0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.505, 11.180]

    # sqrt(Inverse of proportion * 100)  capped (0.5 < X < 5)
    # class_w = [1.430, 5.000, 5.000, 4.226, 5.000, 5.000, 5.000, 5.000, 0.719, 2.377,
    #            0.886, 3.863, 0.869, 1.209, 0.594, 3.780, 1.129, 5.000, 5.000]

    # Do we need to save convergence
    saving = True
    saving_path = None

    # Only train class and center head
    pre_train = False

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
                        help="Index of checkpoint", type=int, default=None)
    parser.add_argument("-s", "--saving_path",
                        help="Path to save checkpoints", default=None)
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

    # Set which gpu is going to be used
    GPU_ID = '0'
    if torch.cuda.device_count() > 1:
        GPU_ID = '0, 1'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    args = parse_args()

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    previous_training_path = args.prev_train_path
    chkp_idx = args.chkp_idx
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join(
            'results', 'checkpoints', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(
            'results', 'checkpoints', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = SemanticKittiConfig()
    if previous_training_path:
        config.load(os.path.join(
            'results', 'checkpoints', previous_training_path))
    config.free_dim = 4
    config.n_frames = 1
    config.reinit_var = True
    config.n_test_frames = 1
    #config.sampling = 'objectness'
    if config.n_frames == 1:
        config.stride = 1
        config.sampling = 'importance'
    else:
        # n_frames==2
        config.stride = 2
        config.sampling = None
    config.decay_sampling = 'None'
    config.big_gpu = True

    config.task_set = args.task_set
    config.saving_path = args.saving_path

    # Initialize datasetss
    test_dataset = SemanticKittiDataset(config, set='validation',
                                        balance_classes=False, seqential_batch=True)

    # Initialize samplers
    test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=0,  # config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values,
                 test_dataset.ignored_labels)

    # Define a trainer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart validation')
    print('**************')

    # Validation
    if config.n_frames > 1:
        tester.panoptic_4d_test(net, test_loader, config)
    else:
        tester.slam_segmentation_test(net, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
