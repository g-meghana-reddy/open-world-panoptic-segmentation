#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis

from PIL import Image
import pdb


def prepare_legend(frame_size):
    legend = Image.open('/project_data/ramanan/achakrav/4D-PLS/outputs/Legend.png')
    w1, h1 = frame_size
    w2, h2 = legend.size
    legend = legend.resize((3*w2, 3*h2))
    img = Image.new('RGB', (w1, 3*h2), (255, 255, 255))
    img.paste(legend, (20, 0)) # (w1-3*w2, 0)) # (w1//2 - 3*w2//2, 0))
    return img


def add_legend(frame, legend):
    img = Image.new('RGB', (frame.width, frame.height+legend.height))
    img.paste(frame, (0, 0))
    img.paste(legend, (0, frame.height))
    return img


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--task', '-t',
      type=int,
      default=0,
      required=False,
      help='Task set to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      default=None,
      required=False,
      help='Alternate location for labels, to use predictions folder. '
      'Must point to directory containing the predictions in the proper format '
      ' (see readme)'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--save-dir', '-sd',
      type=str,
      default=None,
      required=True,
      help='Location to save visualizations.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--visu',
      type=int,
      default=0,
      required=True,
      help='visualization 0: known, 1: unknown, 2: both',
  )
    
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("Predictions", FLAGS.predictions)
  print("Save_dir", FLAGS.save_dir)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("do_instances", FLAGS.do_instances)
  print("ignore_safety", FLAGS.ignore_safety)
  print("offset", FLAGS.offset)
  print("*" * 80)
  
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '2013_05_28_drive_{0:04d}_sync'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, 'data_3d_raw', FLAGS.sequence, 'velodyne_points', 'data/')
  gt_paths = os.path.join(FLAGS.dataset, 'data_3d_raw_labels', FLAGS.sequence, 'labels')
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  file_ids = set(vf[:-6] for vf in os.listdir(gt_paths) if vf.endswith('.label'))
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn if f[:-4] in file_ids]
  scan_names.sort()

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
    if FLAGS.predictions is not None:
      label_paths = os.path.join(FLAGS.predictions, FLAGS.sequence, 'predictions')
    else:
      label_paths = gt_paths

    if os.path.isdir(label_paths):
      print("Labels folder exists! Using labels from %s" % label_paths)
    else:
      print("Labels folder doesn't exist! Exiting...")
      quit()
    # populate the pointclouds
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    # only load .label files and not center annotations
    label_names = [l for l in label_names if l.endswith(".label")]
    label_names.sort()

    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert(len(label_names) == len(scan_names))

  # create a scan
  if FLAGS.ignore_semantics:
    scan = LaserScan(project=True)  # project all opened scans to spheric proj
  else:
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)

    #Meghana
    if FLAGS.visu == 0:
    # 1. Known instances with distinct id colors
        scan = SemLaserScan(nclasses, color_dict, project=True, unknown = False, known = True, distinct = True)

    if FLAGS.visu == 1:
    # 2. Unknown instances with distinct id colors
        scan = SemLaserScan(nclasses, color_dict, project=True, unknown = True, known = False, distinct = True)
    
    if FLAGS.visu == 2:
    # 3. Known and Unknown instances with continous id colors (Green and Red)
        scan = SemLaserScan(nclasses, color_dict, project=True, unknown = True, known = True, distinct = False)

    #Meghana
    
  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  if not semantics:
    label_names = None
  vis = LaserScanVis(scan=scan,
                     scan_names=scan_names,
                     label_names=label_names,
                     offset=FLAGS.offset,
                     semantics=semantics, instances=instances and semantics)

  # print instructions
  # print("To navigate:")
  # print("\tb: back (previous scan)")
  # print("\tn: next (next scan)")
  # print("\tq: quit (exit program)")

  # run the visualizer
  # vis.run()

  # use visualizer to save image
  write_dir = FLAGS.save_dir

  while vis.offset < vis.total:
    img_numpy = vis.canvas.render()
    img_file = '{}.png'.format(vis.offset)
    if FLAGS.predictions is None:
      img_file = os.path.join(write_dir, 'gt', img_file)
    else:
      img_file = os.path.join(write_dir, '{}_pred_TS{}'.format(FLAGS.visu,FLAGS.task), img_file)
    img = Image.fromarray(img_numpy)

    legend = prepare_legend(img.size)
    img = add_legend(img, legend)

    img.save(img_file)
    vis.offset += 1
    vis.update_scan()
    
  vis.offset = 0