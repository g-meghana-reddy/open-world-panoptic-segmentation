#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
#https://github.com/PRBonn/semantic-kitti-api
# Current dir: /project_data/ramanan/achakrav/4D-PLS/utils
# python3 evaluate_segment.py -d ../data/SemanticKitti/ -t 1 -p ../results/validation/TS1 -dc ../data/SemanticKitti/semantic-kitti.yaml -sg ../../hu-segmentation/semantic_kitti_ts1/ -o ../PQ_TS1_unknown
import argparse
import os
import yaml
import sys
import numpy as np
import time
import json

from eval_np import PanopticEval
import matplotlib.pyplot as plt
import pdb


class PanopticSegmentEval(PanopticEval):
  """ Unknown segment evaluation using numpy

  authors: Anirudh Chakravarthy, Meghana Ganesina

  """

  def __init__(self, *args, **kwargs):
    unknown_label = kwargs.pop('unknown_label', 11)
    max_proposals = kwargs.pop('max_proposals', 100)
    super(PanopticSegmentEval, self).__init__(*args, **kwargs)

    self.unknown_label = unknown_label
    self.max_proposals = max_proposals

    self.pan_tp = {i: 0. for i in range(max_proposals)}
    self.pan_fp = {i: 0. for i in range(max_proposals)}
    self.pan_fn = {i: 0. for i in range(max_proposals)}


  def addBatch(self, x_sem, x_inst, y_sem, y_inst, x_obj):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    # self.addBatchSemIoU(x_sem, y_sem)

    # now do the panoptic stuff
    self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst, x_obj)
  
  def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row, num_proposals):
    # make sure instances are not zeros (it messes with my approach)
    x_inst_row = x_inst_row + 1
    y_inst_row = y_inst_row + 1

    # get a class mask
    x_inst_in_cl_mask = x_sem_row == self.unknown_label
    y_inst_in_cl_mask = y_sem_row == self.unknown_label

    # get instance points in class (makes outside stuff 0)
    x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
    y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

    # generate the areas for each unique instance prediction
    unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
    id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
    matched_pred = np.array([False] * unique_pred.shape[0])
    # print("Unique predictions:", unique_pred)

    # generate the areas for each unique instance gt_np
    unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
    id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
    matched_gt = np.array([False] * unique_gt.shape[0])
    # print("Unique ground truth:", unique_gt)

    # generate intersection using offset
    valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
    offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
    unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

    # generate an intersection map
    # count the intersections with over 0.5 IoU as TP
    gt_labels = unique_combo // self.offset
    pred_labels = unique_combo % self.offset
    gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
    pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
    intersections = counts_combo
    unions = gt_areas + pred_areas - intersections
    ious = intersections.astype(np.float) / unions.astype(np.float)


    tp_indexes = ious > 0.5
    self.pan_tp[num_proposals] += np.sum(tp_indexes)
    # self.pan_iou[cl] += np.sum(ious[tp_indexes])

    matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
    matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

    # count the FN
    self.pan_fn[num_proposals] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

    # count the FP
    self.pan_fp[num_proposals] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

    # if num_proposals > 1:
    #   self.pan_tp[num_proposals] += self.pan_tp[num_proposals-1]
    #   self.pan_fn[num_proposals] += self.pan_fn[num_proposals-1]
    #   self.pan_fp[num_proposals] += self.pan_fp[num_proposals-1]


# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_segment.py")
  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions',
      '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset')
  parser.add_argument(
      '--split',
      '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg',
      '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit',
      '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--task_set',
      '-t',
      type=int,
      required=True,
      default=2,
      help='Task set to evaluate'
      ' Defaults to %(default)s',
  )
  parser.add_argument(
      '--min_inst_points',
      type=int,
      required=False,
      default=50,
      help='Lower bound for the number of points to be considered instance',
  )
  parser.add_argument(
      '--output',
      '-o',
      type=str,
      required=False,
      default=None,
      help='Output directory for scores.txt and detailed_results.html.',
  )
  parser.add_argument(
      '--segments',
      '-sg',
      type=str,
      required=False,
      default=None,
      help='Directory with output segments.',
  )

  start_time = time.time()

  FLAGS, unparsed = parser.parse_known_args()

  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("Min instance points: ", FLAGS.min_inst_points)
  print("Output directory", FLAGS.output)
  print("Task set:", FLAGS.task_set)
  print("*" * 80)

  if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)

  # assert split
  assert (FLAGS.split in splits)

  # open data config file
  DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))

  # get number of interest classes, and the label mappings
  # class
  class_remap = DATA["task_set_map"][FLAGS.task_set]["learning_map"]
  class_inv_remap = DATA["task_set_map"][FLAGS.task_set]["learning_map_inv"]
  class_ignore = DATA["task_set_map"][FLAGS.task_set]["learning_ignore"]
  nr_classes = len(class_inv_remap)
  class_strings = DATA["labels"]

  # make lookup table for mapping
  # class
  maxkey = max(class_remap.keys())

  # +100 hack making lut bigger just in case there are unknown labels
  class_lut = np.zeros((maxkey + 100), dtype=np.int32)
  class_lut[list(class_remap.keys())] = list(class_remap.values())

  # class
  ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

  print("Ignoring classes: ", ignore_class)

  if FLAGS.task_set == 0:
    unknown_label = 7
    max_proposals = 200
  elif FLAGS.task_set == 1:
    unknown_label = 11
    max_proposals = 60
  else:
    assert False

  # create evaluator
  # class_evaluator = PanopticSegmentEval(nr_classes, None, ignore_class, min_points=FLAGS.min_inst_points, 
  #                                       unknown_label=unknown_label, max_proposals=max_proposals)
  class_evaluator = PanopticSegmentEval(nr_classes, None, ignore_class, min_points=0,
                                        unknown_label=unknown_label, max_proposals=max_proposals)

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
    # populate the label names
    seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
    # populate the label names
    seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  # get objectness paths
  obj_names = []
  for sequence in test_sequences:
    seq_prefix = '{0:02d}_0'.format(int(sequence))
    # populate the objectness names
    obj_pred_names = sorted([
      os.path.join(FLAGS.segments, fn) for fn in os.listdir(FLAGS.segments)
      if fn.endswith(".npz") and fn.startswith(seq_prefix)
    ])
    obj_names.extend(obj_pred_names)
  # print(obj_names)

  # check that I have the same number of files
  assert (len(label_names) == len(pred_names))
  assert (len(label_names) == len(obj_names))

  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison

  complete = len(label_names)
  count = 0
  percent = 10
  num_gt_proposals = 0
  for label_file, pred_file, obj_file in zip(label_names, pred_names, obj_names):
    count = count + 1
    if 100 * count / complete > percent:
      print("{}% ".format(percent), end="", flush=True)
      percent = percent + 10
    # print("evaluating label ", label_file, "with", pred_file)
    # open label

    label = np.fromfile(label_file, dtype=np.uint32)
    objectness = np.load(obj_file, allow_pickle=True)

    u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_label_inst = label >> 16
    if FLAGS.limit is not None:
      u_label_sem_class = u_label_sem_class[:FLAGS.limit]
      u_label_sem_cat = u_label_sem_cat[:FLAGS.limit]
      u_label_inst = u_label_inst[:FLAGS.limit]
    _, count_inst = np.unique(u_label_inst[u_label_sem_class==unknown_label], return_counts=True)
    num_gt_proposals += count_inst.sum()

    label = np.fromfile(pred_file, dtype=np.uint32)

    u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
    u_pred_inst = label >> 16
    if FLAGS.limit is not None:
      u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
      u_pred_sem_cat = u_pred_sem_cat[:FLAGS.limit]
      u_pred_inst = u_pred_inst[:FLAGS.limit]

    # sort proposals by scores in descending order
    obj_indices = (-objectness['segment_scores']).argsort()
    for num_proposal in range(min(len(obj_indices), max_proposals)):
      curr_points = obj_indices[:num_proposal+1]
      u_pred_inst_prop = -np.ones(u_pred_inst.shape[0], dtype=np.int32)
      for indices in objectness['instances'][curr_points]:
        u_pred_inst_prop[indices] = u_pred_inst[indices]
      class_evaluator.addBatch(u_pred_sem_class, u_pred_inst_prop, u_label_sem_class, u_label_inst, num_proposal)

  print("100%")
  complete_time = time.time() - start_time

  # now make a nice dictionary
  output_dict = {}
  for num_proposal in range(max_proposals):
    # precision = class_evaluator.pan_tp[num_proposal] / np.maximum(
    #   class_evaluator.pan_tp[num_proposal] + class_evaluator.pan_fp[num_proposal], class_evaluator.eps)
    recall = class_evaluator.pan_tp[num_proposal] / np.maximum(
      class_evaluator.pan_tp[num_proposal] + class_evaluator.pan_fn[num_proposal], class_evaluator.eps)
    # recall = class_evaluator.pan_tp[num_proposal].astype(np.double) / num_gt_proposals
    # output_dict['prec@{}'.format(num_proposal+1)] = precision
    output_dict['recall@{}'.format(num_proposal+1)] = recall

  print(output_dict)
  # precisions = [output_dict['prec@{}'.format(i+1)] for i in range(num_proposal)]
  recalls = [output_dict['recall@{}'.format(i+1)] for i in range(num_proposal)]
  plt.xlabel('No. of proposals')
  plt.ylabel('Recall')
  plt.plot(range(len(recalls)), recalls)
  plt.savefig('recall_TS{}.png'.format(FLAGS.task_set))
  # plt.xlabel('Recall')
  # plt.ylabel('Precision')
  # plt.plot(recalls, precisions)
  # plt.savefig('pr_TS{}.png'.format(FLAGS.task_set))
  # pdb.set_trace()

  # when I am done, print the evaluation
#   class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = class_evaluator.getPQ()
#   class_IoU, class_all_IoU = class_evaluator.getSemIoU()

#   # now make a nice dictionary
#   output_dict = {}

#   # make python variables
#   class_PQ = class_PQ.item()
#   class_SQ = class_SQ.item()
#   class_RQ = class_RQ.item()
#   class_all_PQ = class_all_PQ.flatten().tolist()
#   class_all_SQ = class_all_SQ.flatten().tolist()
#   class_all_RQ = class_all_RQ.flatten().tolist()
#   class_IoU = class_IoU.item()
#   class_all_IoU = class_all_IoU.flatten().tolist()

#   # fill in with the raw values
#   # output_dict["raw"] = {}
#   # output_dict["raw"]["class_PQ"] = class_PQ
#   # output_dict["raw"]["class_SQ"] = class_SQ
#   # output_dict["raw"]["class_RQ"] = class_RQ
#   # output_dict["raw"]["class_all_PQ"] = class_all_PQ
#   # output_dict["raw"]["class_all_SQ"] = class_all_SQ
#   # output_dict["raw"]["class_all_RQ"] = class_all_RQ
#   # output_dict["raw"]["class_IoU"] = class_IoU
#   # output_dict["raw"]["class_all_IoU"] = class_all_IoU

#   if FLAGS.task_set == 0:
#     things = ['car', 'person', 'unknown']
#     stuff = ['road', 'building', 'vegetation', 'fence']
#   if FLAGS.task_set == 1:
#     things = ['car', 'person', 'truck', 'unknown']
#     stuff = ['road', 'building', 'vegetation', 'fence', 'sidewalk', 'terrain', 'pole']
#   elif FLAGS.task_set == 2:
#     # things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
#     things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person']
#     stuff = [
#         'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
#         'traffic-sign'
#     ]
#   all_classes = things + stuff

#   # class

#   output_dict["all"] = {}
#   output_dict["all"]["PQ"] = class_PQ
#   output_dict["all"]["SQ"] = class_SQ
#   output_dict["all"]["RQ"] = class_RQ
#   output_dict["all"]["IoU"] = class_IoU

#   classwise_tables = {}
  
#   for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
#     class_str = class_strings[class_inv_remap[idx]]
#     output_dict[class_str] = {}
#     output_dict[class_str]["PQ"] = pq
#     output_dict[class_str]["SQ"] = sq
#     output_dict[class_str]["RQ"] = rq
#     output_dict[class_str]["IoU"] = iou
  
#   PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
#   PQ_known = np.mean([float(output_dict[c]["PQ"]) for c in all_classes if c != 'unknown'])
#   PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
#   RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
#   SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])

#   PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
#   RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
#   SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])

#   PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
#   RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
#   SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])
#   mIoU = output_dict["all"]["IoU"]
#   known_IoU = np.mean([float(output_dict[c]["IoU"]) for c in all_classes if c != 'unknown'])
#   # Ani
#   if FLAGS.task_set != 2:
#     PQ_unknown = output_dict["unknown"]["PQ"]
#     unknown_IoU = output_dict["unknown"]["IoU"]

#   codalab_output = {}
#   codalab_output["pq_mean"] = float(PQ_all)
#   codalab_output["pq_known_mean"] = float(PQ_known)
#   codalab_output["pq_dagger"] = float(PQ_dagger)
#   codalab_output["sq_mean"] = float(SQ_all)
#   codalab_output["rq_mean"] = float(RQ_all)
#   codalab_output["iou_mean"] = float(mIoU)
#   codalab_output["pq_stuff"] = float(PQ_stuff)
#   codalab_output["rq_stuff"] = float(RQ_stuff)
#   codalab_output["sq_stuff"] = float(SQ_stuff)
#   codalab_output["pq_things"] = float(PQ_things)
#   codalab_output["rq_things"] = float(RQ_things)
#   codalab_output["sq_things"] = float(SQ_things)
#   codalab_output["known_IoU"] = float(known_IoU)
#   # Ani
#   if FLAGS.task_set != 2:
#     codalab_output["pq_unknown_mean"] = float(PQ_unknown)
#     codalab_output["unknown_IoU"] = float(unknown_IoU)

#   print("Completed in {} s".format(complete_time))

#   if FLAGS.output is not None:
#     table = []
#     for cl in all_classes:
#       entry = output_dict[cl]
#       table.append({
#           "class": cl,
#           "pq": "{:.3}".format(entry["PQ"]),
#           "sq": "{:.3}".format(entry["SQ"]),
#           "rq": "{:.3}".format(entry["RQ"]),
#           "iou": "{:.3}".format(entry["IoU"])
#       })

#     print("Generating output files.")
#     # save to yaml
#     output_filename = os.path.join(FLAGS.output, 'scores.txt')
#     with open(output_filename, 'w') as outfile:
#       yaml.dump(codalab_output, outfile, default_flow_style=False)

#     ## producing a detailed result page.
#     output_filename = os.path.join(FLAGS.output, "detailed_results.html")
#     with open(output_filename, "w") as html_file:
#       html_file.write("""
# <!doctype html>
# <html lang="en" style="scroll-behavior: smooth;">
# <head>
#   <script src='https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/js/tabulator.min.js'></script>
#   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabulator/4.4.3/css/bulma/tabulator_bulma.min.css">
# </head>
# <body>
#   <div id="classwise_results"></div>

# <script>
#   let table_data = """ + json.dumps(table) + """


#   table = new Tabulator("#classwise_results", {
#     layout: "fitData",
#     data: table_data,
#     columns: [{title: "Class", field:"class", width:200}, 
#               {title: "PQ", field:"pq", width:100, align: "center"},
#               {title: "SQ", field:"sq", width:100, align: "center"},
#               {title: "RQ", field:"rq", width:100, align: "center"},
#               {title: "IoU", field:"iou", width:100, align: "center"}]
#   });
# </script>
# </body>
# </html>""")
