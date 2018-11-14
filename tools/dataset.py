from collections import Sized

import numpy as np
import sigpy as sp

import random
import re
import os


class DataSet(Sized):

    def __init__(self, directory, namepattern, fileloader, data_key=None, label_key=None,
                 scale=False):
        """

        Args:
            directory (str):        Directory to traverse to find data files
            namepattern (str):      Pattern (regex) for file names of data files to open
            fileloader (callable):  Function that reads the file
            data_key (object):      If loaded data is a dictinoary, extract this feature as data
            label_key (object):     If loaded data is a dictinoary, extract this feature as key
            scale (bool):           Scale data to be in [0, 1]
        """
        self.fileloader = fileloader
        self.data_key = data_key
        self.label_key = label_key
        self.scale = scale

        self.data_files = []
        self.index = 0

        self.epoch = 0

        self._traverse_dir_and_queue_files(directory, namepattern)

        self.next_batch = None
        self.next_batch_sampled = None
        self.next_batch_labels = None


    def _traverse_dir_and_queue_files(self, directory, namepattern):
        # Compile regex for faster searching
        re_pattern = re.compile(namepattern)

        # Traverse given dir, and queue all files matching pattern
        for dirName, subdirList, fileList in os.walk(directory):
            for fname in fileList:
                if re.fullmatch(re_pattern, fname) != None:
                    self.data_files.append(dirName + "/" + fname)


    def _get_next(self):
        if self.index < len(self.data_files):
            self.index += 1
            return self.data_files[self.index - 1]

        else:
            random.shuffle(self.data_files)
            self.epoch += 1
            self.index = 0
            return self.data_files[self.index]


    def ready_next_batch(self, size):
        # If a new batch is already loaded, return
        if self.next_batch is not None: return self.next_batch.shape[0]

        data_list = [None] * size
        label_list = [None] * size

        # Read files and store temporarily
        for i in range(size):
            local_raw_data = self.fileloader(self._get_next())

            if self.data_key is not None:
                local_data = local_raw_data[self.data_key]
            if self.label_key is not None:
                label_list[i] = local_raw_data[self.label_key]

            # Map to [0, 1]
            if self.scale:
                data_list[i] = local_data / np.max(np.abs(local_data))
            else:
                data_list[i] = local_data

        # Store array
        next_batch = np.array(data_list)
        self.next_batch = next_batch

        if self.label_key is not None:
            next_labels = np.array(label_list)
            self.next_batch_labels = next_labels


    def _fix_dimensions(self, x):
        if len(self.next_batch.shape) == 3:
            return np.expand_dims(x, -1)
        else:
            return x


    def get_next_batch(self):
        temp_batch = self._fix_dimensions(self.next_batch)
        self.next_batch = None

        return_this = [temp_batch]

        if self.next_batch_sampled is not None:
            temp_sampled = self._fix_dimensions(self.next_batch)
            self.next_batch_sampled = None

            return_this.append(temp_sampled)

        if self.next_batch_labels is not None:
            temp_labels = self.next_batch_labels
            self.next_batch_labels = None

            return_this.append(temp_labels)

        return return_this


    def sample_loaded_batch(self, operator):
        data_list = [None] * len(self.next_batch)

        for i in range(len(data_list)):
            data_list[i] = operator(self.next_batch[i])

        self.next_batch_sampled = np.array(data_list, dtype=np.complex64)


    def __len__(self):
        return len(self.data_files)
