import argparse
import signal
import sys

sys.path.append('../../..')

from datasets.SemanticKitti import *
from models.architectures import KPFCNN
from train_SemanticKitti import SemanticKittiConfig
from segment_utils import *

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

DATA_DIR = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_set", help="Task Set ID", type=int, default=-1)
    parser.add_argument("-p", "--prev_train_path", help="Directory to load checkpoint", default='4DPLS_original_params_original_repo_nframes1_1e-3')
    parser.add_argument("-i", "--chkp_idx", help="Index of checkpoint", type=int, default=None)
    parser.add_argument("-g", "--gpu-id", help="GPU ID", type=int, default=0)
    args = parser.parse_args()
    return args

def evaluation_4dpls(net, test_loader, config, num_votes=100, chkp_path=None, on_gpu=True):
    '''Evaluation run on 4DPLS'''
    #****************************************************
    # Assigning the device configuration
    #****************************************************

    # Choose to train on CPU or GPU
    if on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    net.to(device)

    #****************************************************
    # Loading the model checkpoint
    #****************************************************

    checkpoint = torch.load(chkp_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    print('**********************************')
    print('Model and training state restored ')
    print('Eval mode ')
    print('**********************************')

    #****************************************************
    # Setting the evaluation config
    #****************************************************

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    test_smooth = 0.5
    last_min = -0.5
    softmax = torch.nn.Softmax(1)
    
    # Number of classes including ignored labels
    nc_model = net.C

    #****************************************************
    # Setting the dataset destination path
    #****************************************************

    # dataset saving path
    data_path = None
    if config.saving:
        data_path = join("/project_data/ramanan/achakrav/4D-PLS/data/segment_dataset")
        if not exists(data_path):
            makedirs(data_path)
    
    for seq in test_loader.dataset.sequences:
        folder = join(data_path, seq)
        if not exists(folder):
            makedirs(folder)

    #****************************************************
    # Starting evaluation: Forward pass
    #****************************************************
    print('**********************************')
    print('Starting evaluation: Forward pass ')
    print('**********************************')

    test_epoch = 0

    # Start test loop
    while True:
        for idx, batch in enumerate(test_loader):

            if idx % 10 != 0:
                continue

            if 'cuda' in device.type:
                batch.to(device)

            with torch.no_grad():
                # perform forward pass in evaluation mode
                outputs, centers_output, _, embedding = net(batch, config)
                probs = softmax(outputs).cpu().detach().numpy()

                # add the column corresponding the zero label to the probabilities
                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                    if label_value in test_loader.dataset.ignored_labels:
                        probs = np.insert(probs, l_ind, 0, axis=1)

                # compute the predictions by finding the argmax of softmax scores
                preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                preds = torch.from_numpy(preds)
                preds.to(outputs.device)
                
            # Get probs, masks, semantic/instance labels, centerness scores, embeddings
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            xyz_points = batch.points
            ins_labels_list = batch.val_ins_labels
            centers_output = centers_output.cpu().detach().numpy()
            embedding = embedding.cpu().detach().numpy()
            
            torch.cuda.synchronize(device)

            # *******************************************************
            # Find the predictions per scan in the batch and save
            # them in scan wise files or it is used to compute the 
            # segments per scan and dataset is generated
            # *******************************************************

            i0 = 0
            for b_i, length in enumerate(lengths):
                # reproj indices
                proj_inds = r_inds_list[b_i]
                # reproj mask
                proj_mask = r_mask_list[b_i]
                # sequence index
                s_ind = f_inds[b_i, 0]
                # frame index
                f_ind = f_inds[b_i, 1]
                # semantic labels
                frame_labels = labels_list[b_i]
                # instance labels
                frame_ins_labels = ins_labels_list
                # softmax outputs
                probs = stk_probs[i0:i0 + length]
                # centerness outputs
                center_props = centers_output[i0:i0 + length]
                # xyz points w.r.t first frame
                frame_xyz_points = xyz_points[b_i].cpu().detach().numpy()
                # embedding outputs
                emb = embedding[i0:i0 + length]

                # Project predictions on the frame points
                # projected softmax outputs
                proj_probs = probs[proj_inds]
                # projected centerness scores
                proj_center_probs = center_props[proj_inds]
                # projected embedding outputs
                proj_emb = emb[proj_inds]
                # projected labels
                proj_labels = frame_labels[proj_inds]
                proj_ins_labels = frame_ins_labels[proj_inds]
                # projected xyz points
                proj_xyz_points = frame_xyz_points[proj_inds]
                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)
                    proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                    proj_center_probs = np.expand_dims(proj_center_probs, 0)
                    proj_emb = np.expand_dims(proj_emb, 0)

                # Initialize the projected frame wise predictions
                frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                frame_probs_softmax = np.zeros((proj_mask.shape[0], nc_model))
                frame_gt_labels =  np.zeros((proj_mask.shape[0]))
                frame_gt_ins_labels = np.zeros((proj_mask.shape[0]))
                frame_center_preds = np.zeros((proj_mask.shape[0]))
                frame_emb_preds = np.zeros((proj_mask.shape[0], config.first_features_dim), dtype=np.float32)
                frame_xyz = np.zeros((proj_mask.shape[0], 3), dtype=np.float32)
                frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                # Insert false columns for ignored labels
                frame_probs_uint8_bis = frame_probs_uint8.copy()
                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                    if label_value in test_loader.dataset.ignored_labels:
                        frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                # Compute the final frame wise projections
                frame_gt_labels[proj_mask] = proj_labels
                frame_gt_ins_labels[proj_mask] = proj_ins_labels
                frame_center_preds[proj_mask] = proj_center_probs[:, 0]
                frame_emb_preds[proj_mask] = proj_emb
                frame_probs_softmax[proj_mask, :] = proj_probs
                frame_xyz[proj_mask, :] = proj_xyz_points
                frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,axis=1)].astype(np.int32)

                seq_name = test_loader.dataset.sequences[s_ind]
                frame_name = test_loader.dataset.frames[s_ind][f_ind]
                filepath = join(data_path, seq_name, frame_name)
                scan_file = join(DATA_DIR, 'sequences', seq_name, 'velodyne', frame_name + '.bin')
                
                # Construct the hierarchical tree per scan for all the thing class points and 
                # compute scores per segment to generate the segment dataset
                print("*************** Started constructing tree scan_file: {}_{}  *************** ".format(seq_name, frame_name))
                # generate_segments_per_scan(scan_file, frame_emb_preds, frame_gt_labels, frame_gt_ins_labels, filepath) 
                generate_segments_per_scan(
                    scan_file, frame_emb_preds, frame_preds, frame_gt_labels,
                    frame_gt_ins_labels, frame_xyz, filepath
                )
                print("*************** Ended constructing tree scan_file: {}_{}    *************** ".format(seq_name, frame_name))
                i0 += length


        # Update minimum od potentials
        new_min = torch.min(test_loader.dataset.potentials)
        print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

        if last_min + 1 < new_min:
            # Update last_min
            last_min += 1

        test_epoch += 1

        # Break when reaching number of desired votes
        if last_min > num_votes:
            break

    return  
    
def generate_segments_per_scan(scan_file, frame_emb_preds, frame_pred_labels, frame_gt_labels, frame_gt_ins_labels, frame_xyz, filepath):
    '''Constructs the hierarchical tree using the thing class points and per 
        node/segment score is computed to generate the segment dataset.'''
    #****************************************************
    # Load velodyne points
    #****************************************************
    pts_velo_cs = load_vertex(scan_file)
    pts_indexes = np.arange(len(pts_velo_cs))

    #********************************************************************
    # Compute the things mask and compute labels, xyz points accordingly
    #********************************************************************
    # thing classes: [1,2,3,4,5,6,7,8]
    # things_mask = np.where(np.logical_and(frame_gt_labels > 0 , frame_gt_labels < 9))
    things_mask = np.where(np.logical_and(frame_pred_labels > 0 , frame_pred_labels < 9))
    gt_mask = np.where(np.logical_and(frame_gt_labels > 0 , frame_gt_labels < 9))


    # generate all labels for things only
    first_frame_coordinates = frame_xyz[things_mask]
    pts_velo_cs_objects = pts_velo_cs[things_mask]
    pts_indexes_objects = pts_indexes[things_mask]
    pts_embeddings_objects = frame_emb_preds[things_mask]
    gt_semantic_labels = frame_gt_labels[things_mask]

    file = scan_file.split('.')[0]
    gt_file = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/sequences/08/labels/{}.label'.format(file)
    config = "/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti-orig.yaml"
    with open(config, 'r') as stream:
        doc = yaml.safe_load(stream)
        all_labels = doc['labels']
        learning_map_inv = doc['learning_map_inv']
        learning_map_doc = doc['learning_map']
        learning_map = np.zeros((np.max([k for k in learning_map_doc.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map_doc.items():
            learning_map[k] = v

        inv_learning_map = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), 
                            dtype=np.int32)
        for k, v in learning_map_inv.items():
            inv_learning_map[k] = v

    gt_label = np.fromfile(gt_file, dtype=np.int32)
    sem_gt = learning_map[gt_label & 0xFFFF]
    ins_gt = gt_label >> 16
    gt_mask = np.where(np.logical_and(sem_gt > 0 , sem_gt < 9))
    gt_instance_ids = ins_gt[gt_mask]
    gt_instance_indexes = np.arange(ins_gt.shape[0])[gt_mask]
    gt_instance_ids_objects = ins_gt[gt_mask]
    gt_instance_indexes_objects = np.arange(ins_gt.shape[0])[gt_mask]

    if len(pts_velo_cs_objects) < 1:
        return
    
    # Define the hierarchical DBSCAN thresholds
    eps_list_tum = [1.2488, 0.8136, 0.6952, 0.594, 0.4353, 0.3221]

    #********************************************************************
    # Initialize the TreeSegment class for the current scan
    #********************************************************************
    original_indices = np.arange(pts_velo_cs_objects.shape[0])
    segment_tree = TreeSegment(original_indices, 0)
    
    #********************************************************************
    # Compute the hierarchical tree of segments
    #********************************************************************
    segment_tree.child_segments = compute_hierarchical_tree(eps_list_tum, pts_velo_cs_objects, pts_indexes_objects, gt_instance_indexes_objects, gt_instance_ids_objects, original_indices)

    #********************************************************************
    # Traverse the computed hierarchical tree to store the segments
    #********************************************************************
    segment_tree_traverse(segment_tree, pts_embeddings_objects, pts_velo_cs_objects, gt_semantic_labels, first_frame_coordinates, filepath, 0, set())
    
    return

if __name__ == '__main__':

    #****************************************************
    # Step 1: Initialize the environment
    #****************************************************
    print('*****************************')
    print('Initializing the environment ')
    print('*****************************')

    # Get the command line arguments
    args = parse_args()

    # Set which gpu is going to be used
    GPU_ID = str(args.gpu_id)
    # if torch.cuda.device_count() > 1:
    #     GPU_ID = '0, 1'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    #****************************************************
    # Step 2: Previous chkp
    #****************************************************
    print('************************')
    print('Choosing the checkpoint ')
    print('************************')

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    previous_training_path = args.prev_train_path
    # previous_training_path = "/project_data/ramanan/mganesin/4D-PLS/results"
    chkp_idx = args.chkp_idx
    if previous_training_path:
        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join("/project_data/ramanan/mganesin/4D-PLS/results", previous_training_path, "checkpoints")
        # chkp_path = os.path.join('../../','results/checkpoints', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(chkp_path, chosen_chkp)
    else:
        chosen_chkp = None

    #****************************************************
    # Step 3: Loading configuration
    #****************************************************
    print('**********************')
    print('Loading configuration ')
    print('**********************')

    # Initialize configuration class from checkpoint path
    config = SemanticKittiConfig()
    if previous_training_path:
        config.load(os.path.join('/project_data/ramanan/mganesin/4D-PLS/results', previous_training_path))
        # config.load(os.path.join('../', 'results', previous_training_path))

    # Set the other configuration parameters to match the training config
    config.free_dim = 4
    config.n_frames = 1
    config.reinit_var = True
    config.n_test_frames = 1
    if config.n_frames == 1:
        config.stride = 1
        config.sampling = 'importance'
    else:
        config.stride = 2
        config.sampling = None
    config.decay_sampling = 'None'
    config.big_gpu = True
    config.task_set = args.task_set

    #****************************************************
    # Step 4: Loading datasets
    #****************************************************
    print('*********************************************')
    print('Loading datasets, dataloaders, data samplers ')
    print('*********************************************')

    # Initialize datasets
    # training_dataset = SemanticKittiDataset(config, set='training',
    #                                           balance_classes=False, datapath=DATA_DIR)
    test_dataset = SemanticKittiDataset(config, set='validation',
                                        balance_classes=False, seqential_batch=True, datapath=DATA_DIR)

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
                             num_workers=0, #config.input_threads,
                             pin_memory=True)

    #****************************************************
    # Step 5: Loading Calibration
    #****************************************************
    print('********************')
    print('Loading calibration ')
    print('********************')

    # Calibrate max_in_point value
    # training_sampler.calib_max_in(config, training_loader, verbose=True)
    test_sampler.calib_max_in(config, test_loader, verbose=True)

    # Calibrate samplers
    # training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    #****************************************************
    # Step 6: Defining Model Instnace
    #****************************************************
    print('************************')
    print('Defining model instance ')
    print('************************')

    # Define network model
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    #****************************************************
    # Step 7: Run evaluation on the dataset
    #****************************************************
    print('*******************')
    print('Running evaluation ')
    print('*******************')

    evaluation_4dpls(net, test_loader, config, chkp_path=chosen_chkp)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
