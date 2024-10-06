import joblib
import numpy as np
import os
import torch
from time import time
from torch.utils.data import Dataset, DataLoader

from simclr.sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


class HARDataset(Dataset):
    def __init__(self, args, phase):
        self.filename = os.path.join(args['root_dir'], args['data_file'])

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print('The data is not available. '
                  'Ensure that the data is present in the directory.')
            exit(0)

        # Loading the data
        self.data_raw = self.load_dataset(self.filename)
        assert args['input_size'] == self.data_raw[phase]['data'].shape[1]

        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               args['window'],
                               args['overlap'])

        print('The dataset is: {}. The phase is: {}. The size of the dataset '
              'is: {}'.format(args['dataset'], phase, self.data.shape))

    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]
        data = torch.from_numpy(data)

        label = torch.from_numpy(np.asarray(self.labels[index]))
        return data, label


def load_har_dataset(args, pretrain=True):
    batch_size = args['batch_size'] if pretrain else args[
        'classifier_batch_size']

    datasets = {x: HARDataset(args=args, phase=x) for x in
                ['train', 'val', 'test']}
    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=batch_size,
                                  shuffle=True if x == 'train' else False,
                                  num_workers=0, pin_memory=True) for x in
                    ['train', 'val', 'test']}

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return data_loaders, dataset_sizes
