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
import argparse
import signal
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from datasets.SemanticKitti import *
from models.architectures import KPFCNN
from utils.config import Config
from utils.evaluate_confusion import confusion_openset
from utils.metrics import fast_confusion
from utils.trainer import ModelTrainer

import sklearn
from tqdm import tqdm
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
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
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
    n_frames = 1 # 4
    max_in_points = 100000
    max_val_points = 100000

    # Number of batch
    batch_num = 4
    val_batch_num = 1

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.06 * 2

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
    in_features_dim = 3
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
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

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
    validation_size = 4071

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
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=2)
    parser.add_argument("-p", "--prev_train_path", help="Directory to load checkpoint", default=None)
    parser.add_argument("-i", "--chkp_idx", help="Index of checkpoint",  default=None)
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

    # previous_training_path = 'Log_2020-06-05_17-18-35'
    # previous_training_path = 'Log_2020-10-06_16-51-05'#'Log_2020-08-30_01-29-20'
    # previous_training_path = ''
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    # chkp_idx = None
    previous_training_path = args.prev_train_path
    chkp_idx = args.chkp_idx

    # Find all snapshot in the chosen training folder
    chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    ##############
    # Prepare Data
    ##############


    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = SemanticKittiConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None
    config.pre_train = False # True
    config.free_dim = 4
    config.n_frames = 1 # 2
    config.reinit_var = True
    config.n_test_frames = 1
    config.stride = 1
    #config.sampling = 'objectness'
    config.sampling = 'importance'
    config.decay_sampling = 'None'
    config.validation_size = 4071

    config.task_set = args.task_set
    
    if config.task_set in [0,1]:
        return_unknowns = True
    else:
        return_unknowns = False

    # Initialize datasets
    # training_dataset = SemanticKittiDataset(config, set='training',
    #                                         balance_classes=True)
    test_dataset = SemanticKittiDataset(config, set='validation',
                                        balance_classes=False,
                                        return_unknowns=return_unknowns,seqential_batch=True)

    # Initialize samplers
    # training_sampler = SemanticKittiSampler(training_dataset)
    test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    # training_loader = DataLoader(training_dataset,
    #                              batch_size=1,
    #                              sampler=training_sampler,
    #                              collate_fn=SemanticKittiCollate,
    #                              num_workers=config.input_threads,
    #                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate max_in_point value
    # training_sampler.calib_max_in(config, training_loader, verbose=True)
    test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    # training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_class_w(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()

    checkpoint = torch.load(chosen_chkp) #, map_location=torch.cuda.current_device())
    
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    net.load_state_dict(checkpoint['model_state_dict'])
    # self.epoch = checkpoint['epoch']
    net.eval()

    # Define a trainer class
    # trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart forward pass')
    print('**************')
    
    softmax = torch.nn.Softmax(1)
    predictions = []
    true_mapped_labels = []
    true_unmapped_labels = []

    val_label_values = test_loader.dataset.label_values
    
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
        if config.task_set in [0,1]:
            unknown_labels_list = batch.val_unk_labels_list
            unknown_label_values = list(test_dataset.unknown_label_to_names.keys())

        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            frame_labels = labels_list[b_i]
            if config.task_set in [0,1]:
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
            if config.task_set in [0,1]:
                true_unmapped_labels += [frame_unknown_labels[proj_mask]]

    print('\nCreate confusion matrix')
    print('**************')

    if args.task_set == 0:
        k = 6
    elif args.task_set == 1:
        k = 10
    else:
        k = 19
        
    if args.task_set == 2:
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
    
    if args.task_set != 2:
        conf_matrix_2 = np.delete(conf_matrix_2, 0, axis=0)
        conf_matrix_2 = np.delete(conf_matrix_2, 0, axis=1)

    # Balance with real validation proportions
    if args.task_set == 2:
        conf_matrix_1 = conf_matrix_1.T
        conf_matrix_1 = conf_matrix_1.astype(np.float64)
        conf_matrix_1 /= np.expand_dims((np.sum(conf_matrix_1, axis=1) + 1e-6), 0)
        y_labels = np.array(test_dataset.label_names)[1:]
        x_labels = y_labels
        
        plt.figure(figsize = (30,10))
        sns.heatmap(conf_matrix_1, xticklabels=x_labels, yticklabels=y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig('confusion_matrix_ts{}_balanced.png'.format(args.task_set))
    
    else:
        
        # Meghs
        
        # Unknown to known confusion
        conf_matrix_1 = conf_matrix_1.T
        unk_to_known_conf = np.zeros(conf_matrix_2.shape)
        unk_to_known_conf[:, k:] = conf_matrix_2[:, k:]
        unk_to_known_conf[:, :k] = conf_matrix_1[:-1, :k]
        unk_to_known_conf /= np.expand_dims(np.sum(unk_to_known_conf, axis = 0)+ 1e-6, 0)
        
        unk_to_known_y_labels = np.array(test_dataset.label_names)[1:-1]
        unk_to_known_x_labels = np.concatenate([unk_to_known_y_labels, test_dataset.unknown_label_names])
        
        plt.figure(figsize = (20,10))
        sns.heatmap(unk_to_known_conf, xticklabels=unk_to_known_x_labels, yticklabels=unk_to_known_y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig('extended_confusion_matrix_ts{}_balanced.png'.format(args.task_set))
        
        # Known to Unknown confusion
        conf_matrix = conf_matrix_1.astype(np.float64)
        conf_matrix /= np.expand_dims(np.sum(conf_matrix, axis=0)+ 1e-6, 0)
        y_labels = np.array(test_dataset.label_names)[1:]
        x_labels = y_labels
        
        plt.figure(figsize = (20,10))
        sns.heatmap(conf_matrix, xticklabels=x_labels, yticklabels=y_labels, cmap='Blues', robust=True, square=True)
        plt.xlabel('Groundtruth Class')
        plt.ylabel('Detected Class')
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        plt.savefig('normal_confusion_matrix_ts{}_balanced.png'.format(args.task_set))

        # Meghs
    