import numpy as np
import yaml
import os
import glob
import concurrent.futures

MAX_PROCESSOR = 8
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSOR)
jobs = []

task_set = 0
config_file = '/project_data/ramanan/achakrav/4D-PLS/data/SemanticKitti/semantic-kitti.yaml'
log_dir = '/project_data/ramanan/mganesin/4D-PLS/results_baseline/validation/TS{}'.format(task_set)
prediction_path = '/project_data/ramanan/mganesin/4D-PLS/test_baseline/val_preds_TS{}/val_probs'.format(task_set)
save_path = log_dir
data_dir = '../data/SemanticKitti/'

on_val = True
tracked_unknown = False
baseline = True
if task_set < 2 and not baseline:
    if tracked_unknown:
        ins_ext = 't'
    else:
        ins_ext = 'u'
else:
    ins_ext = 'i'
ins_ext = 'i'

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path+'/sequences'):
    os.makedirs(save_path+'/sequences')

save_path = save_path + '/sequences'

if task_set == 0:
    sem = [0,3,4,5,6]
    inst = [1,2,7]
elif task_set == 1:
    sem = [0,4,5,6,7,8,9,10]
    inst = [1,2,3,11]
else:
    sem = range(7,18)
    inst = range(0,7)

def write_pred(prediction_path, save_path, sequence, scene, inv_learning_map):

    sem_preds = np.load('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene))
    ins_preds = np.load('{}/{:02d}_{:07d}_{}.npy'.format(prediction_path, sequence, scene, ins_ext))

    ins_preds = ins_preds.astype(np.int32)
    for idx, semins in enumerate(np.unique(sem_preds)):
        #Meghs
        #if semins < 1 or semins > 8:
        if semins in sem:
            valid_ind = np.argwhere((sem_preds == semins) & (ins_preds == 0))[:, 0]
            ins_preds[valid_ind] = semins

    for idx, semins in enumerate(np.unique(ins_preds)):

        valid_ind = np.argwhere(ins_preds == semins)[:, 0]
        # ins_preds[valid_ind] = 20 + semins
        if valid_ind.shape[0] < 25:
            ins_preds[valid_ind] = 0

    new_preds = np.left_shift(ins_preds, 16)
    inv_sem_labels = inv_learning_map[sem_preds]
    new_preds = np.bitwise_or(new_preds, inv_sem_labels)

    new_preds.tofile('{}/{:02d}/predictions/{:06d}.label'.format(save_path, sequence, scene))
    return True


with open(config_file, 'r') as stream:
    doc = yaml.safe_load(stream)
    learning_map_doc = doc['task_set_map'][task_set]['learning_map']
    inv_learning_map_doc = doc['task_set_map'][task_set]['learning_map_inv']

inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), dtype=np.int32)
for k, v in inv_learning_map_doc.items():
    inv_learning_map[k] = v

sequences = [8]
if not on_val:
    sequences = [11,12,13,14,15,16,17,18,19,20,21]

for sequence in sequences:
    if not os.path.exists('{}/{:02d}'.format(save_path, sequence)):
        os.makedirs('{}/{:02d}'.format(save_path, sequence))
    if not os.path.exists('{}/{:02d}/predictions'.format(save_path, sequence)):
        os.makedirs('{}/{:02d}/predictions'.format(save_path, sequence))

    n_scenes = len(glob.glob('{}/sequences/{:02d}/velodyne/*.bin'.format(data_dir, sequence)))
    print ('Processing scene {}'.format(sequence))
    for scene in range(0, n_scenes):
        if not os.path.exists('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene)):
            print('Scene : {} is missing'.format(scene))
            print ('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene))
            continue
        #write_pred(prediction_path, save_path, sequence, scene, inv_learning_map)
        jobs.append(executor.submit(write_pred, prediction_path, save_path, sequence, scene, inv_learning_map))

    completed = 0
    for future in concurrent.futures.as_completed(jobs):
        completed+=1
        ret = future.result()
        print('{}/{}'.format(completed, n_scenes), end="\r")

executor.shutdown(wait=True)
