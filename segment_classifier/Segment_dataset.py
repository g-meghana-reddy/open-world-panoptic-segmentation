import os
import json
import numpy as np


class Segment_dataset():

    def __init__(self, dataset_path, split= 'training'):
        self.path = dataset_path
        self.split = split
        self.files = self.load_paths(self.split)
        

    def load_paths(self, split):
        sequences = []
        for seq in os.listdir(self.path):
            if split == 'training' and seq != '08':
                sequences.append(os.path.join(self.path, seq))
            elif split == 'validation' and seq == '08':
                sequences.append(os.path.join(self.path, seq))

        paths = [os.path.join(sequence, scans) for sequence in sequences for scans in os.listdir(sequence) ]
        
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

        # Perform any data augmentation ?
        return segment_data


