from collections import Sized

import numpy as np
import sigpy as sp

import random
import re
import os
import sys


class DataSet(Sized):

    def __init__(self, directory, namepattern, fileloader, data_key=None, label_key=None,
                 scale=False, cache=False, augment=False, crop=None, test_rate=0):
        """

        Args:
            directory (str):        Directory to traverse to find data files
            namepattern (str):      Pattern (regex) for file names of data files to open
            fileloader (callable):  Function that reads the file
            data_key (object):      If loaded data is a dictinoary, extract this feature as data
            label_key (object):     If loaded data is a dictinoary, extract this feature as key
            scale (bool):           Scale data to be in [0, 1]
            cache (bool):           Cache dataset in RAM. Warning: this may use A LOT of memory
            augment (bool):         Apply data augmentation
            crop (tuple):           Crop all data to be this size (crops from center)
            test_rate (float):      Size of test set relative to training set
        """
        self.fileloader = fileloader
        self.data_key = data_key
        self.label_key = label_key
        self.scale = scale
        self.crop = crop

        self.cache = cache
        self.stop_cache = False
        self.cache_threshold = 512 # mb

        self.augment = augment

        self.data_files = []
        self.cached_files = {}
        self.index = 0
        self.epoch = 0

        self.test_files = []
        self.test_rate = test_rate

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

        random.shuffle(self.data_files)

        for i in range(int(len(self.data_files) * self.test_rate)):
            self.test_files.append(self.data_files.pop())


    def _enough_memory(self):
        if self.stop_cache:
            return False
        else:
            if psutil.virtual_memory().available < self.cache_threshold*1e6:
                print()
                print("Less than {} mb left in memory, DataSet will not cache any more files".format(self.cache_threshold), file=sys.stderr)
                self.stop_cache = True
                return False
            else:
                return True


    def _get_data_sample(self, filename):
        if filename in self.cached_files:
            return self.cached_files[filename]
        else:
            file = self.fileloader(filename)
            if self.cache and self._enough_memory(): self.cached_files[filename] = file
            return file


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
            local_raw_data = self._get_data_sample(self._get_next())

            if self.data_key is not None:
                local_data = local_raw_data[self.data_key]
            else:
                local_data = local_raw_data

            if self.label_key is not None:
                label_list[i] = local_raw_data[self.label_key]

            # Map to [0, 1]
            if self.scale:
                local_data -= np.min(np.abs(local_data))
                local_data /= np.max(np.abs(local_data))

            if self.crop is not None:
                middle = list(map(lambda x: x//2, local_data.shape))
                local_data = local_data[middle[0] - self.crop[0]//2 : middle[0] + self.crop[0]//2, middle[1] - self.crop[1] // 2 : middle[1] + self.crop[1]//2]

            data_list[i] = local_data

        # Store array
        next_batch = np.array(data_list)

        if self.augment:
            augment(next_batch)

        self.next_batch = next_batch

        if self.label_key is not None:
            next_labels = np.array(label_list)
            self.next_batch_labels = next_labels


    def _fix_dimensions(self, x):
        if len(x.shape) == 3:
            return np.expand_dims(x, -1)
        else:
            return x


    def get_next_batch(self):
        temp_batch = self._fix_dimensions(self.next_batch)
        self.next_batch = None

        return_this = [temp_batch]

        if self.next_batch_sampled is not None:
            temp_sampled = self._fix_dimensions(self.next_batch_sampled)
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


    def get_test_set(self, sample_op=None):
        data_list = [None] * len(self.test_files)
        label_list = [None] * len(self.test_files)
        sampled_list = [None] * len(self.test_files)

        for i in range(len(self.test_files)):
            local_raw_data = self.fileloader(self.test_files[i])

            if self.data_key is not None:
                local_data = local_raw_data[self.data_key]
            else:
                local_data = local_raw_data

            if self.label_key is not None:
                label_list[i] = local_raw_data[self.label_key]

            # Map to [0, 1]
            if self.scale:
                local_data -= np.min(np.abs(local_data))
                local_data /= np.max(np.abs(local_data))

            if self.crop is not None:
                middle = list(map(lambda x: x//2, local_data.shape))
                local_data = local_data[middle[0] - self.crop[0]//2 : middle[0] + self.crop[0]//2, middle[1] - self.crop[1] // 2 : middle[1] + self.crop[1]//2]

            data_list[i] = local_data

            if sample_op is not None:
                sampled_list[i] = sample_op(data_list[i])

        return_this = [self._fix_dimensions(np.array(data_list))]

        if sample_op is not None:
            return_this.append(self._fix_dimensions(np.array(sampled_list)))

        if self.label_key is not None:
            return_this.append(self._fix_dimensions(np.array(label_list)))

        return return_this


    def __len__(self):
        return len(self.data_files)


def augment(data):
    flip_these = np.random.permutation(len(data))[0:np.random.randint(0, len(data))]
    add_noise = np.random.permutation(len(data))[0:np.random.randint(0, int(math.ceil(len(data) / 2)))]

    if len(data.shape) == 4:
        for ind in flip_these:
            data[ind, :, :, :] = np.flip(data[ind, :, :, :], axis=np.random.randint(2))
        for ind in add_noise:
            data[ind, :, :, :] += np.random.normal(0, 0.02, data[ind].shape)
    elif len(data.shape) == 3:
        for ind in flip_these:
            data[ind, :, :] = np.flip(data[ind, :, :], axis=np.random.randint(2))
        for ind in add_noise:
            data[ind, :, :] += np.random.normal(0, 0.02, data[ind].shape)
    else:
        raise ValueError("data has shape {}, which is unsupported".format(data.shape))

