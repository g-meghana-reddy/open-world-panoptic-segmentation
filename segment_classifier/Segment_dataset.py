import os
import json
import numpy as np


class Segment_dataset():

    def __init__(self, dataset_path):
        self.path = dataset_path
        self.files = self.load_paths()

    def load_paths(self):
        paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.path)) for f in fn]
        paths.sort()
        return np.array(paths)

    def __len__(self):
        """
        Return the length of data here
        """
        return self.files.shape[0]

    def __getitem__(self, index):
        file = self.files[index]
        segment_data = json.load(open(file))

        # Perfoem any data augmentation ?
        return segment_data


