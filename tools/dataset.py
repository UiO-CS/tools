import numpy as np
import sigpy as sp

import random
import re
import os


class DataSet:

    def __init__(self, directory, namepattern, fileloader, data_key=None):
        self.fileloader = fileloader
        self.data_key = data_key

        self.data_files = []
        self.index = 0

        self.epoch = 0

        self._traverse_dir_and_queue_files(directory, namepattern)

        self.next_batch = None
        self.next_batch_sampled = None


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

        # Read files and store temporarily
        for i in range(size):
            local_data = self.fileloader(self._get_next())

            if self.data_key is not None:
                local_data = local_data[self.data_key]

            data_list[i] = local_data / np.max(np.abs(local_data)) * 256

        # Store array
        next_batch = np.array(data_list)
        self.next_batch = next_batch


    def get_next_batch(self):
        temp_batch = np.expand_dims(self.next_batch, -1)
        self.next_batch = None

        if self.next_batch_sampled is not None:
            temp_sampled = np.expand_dims(self.next_batch_sampled, -1)
            self.next_batch_sampled = None

            return temp_batch, temp_sampled

        else:
            return temp_batch


    def sample_loaded_batch(self, operator):
        data_list = [None] * len(self.next_batch)

        for i in range(len(data_list)):
            data_list[i] = operator(self.next_batch[i])

        self.next_batch_sampled = np.array(data_list, dtype=np.complex64)
