import argparse
from datetime import date

import numpy as np
import os
import pandas as pd
import random
from pickle import dump

# For data preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Setting seeds
np.random.seed(42)
random.seed(42)


SENSORS = ['acc_x', 'acc_y', 'acc_z']


def load_args():
    print('Parameters for preparing Motionsense')
    dataset_loc = 'data'

    args = {'dataset_loc': dataset_loc,
            'original_sampling_rate': 50,
            'sampling_rate': 50}
    return args


# from the dataset repo
def map_activity_to_id():
    action_to_id = {"dws": 0, "ups": 1, "wlk": 2, "jog": 3, "std": 4, "sit": 5}
    folder_to_action = {'dws_1': 0, 'dws_11': 0, 'dws_2': 0, 'jog_16': 3,
                        'jog_9': 3, 'sit_13': 5, 'sit_5': 5,
                        'std_14': 4, 'std_6': 4, 'ups_12': 1, 'ups_3': 1,
                        'ups_4': 1, 'wlk_15': 2, 'wlk_7': 2,
                        'wlk_8': 2}

    return action_to_id, folder_to_action


def read_data(current, acc_folder, args):
    all_data = np.zeros((1, 4))

    for subj in range(1, 25):
        acc = os.path.join(args['root_folder'], acc_folder, current,
                           'sub_' + str(subj) + '.csv')
        acc_data = pd.read_csv(acc)

        # Dropping the first column
        acc_data = acc_data.drop(['Unnamed: 0'], axis=1)
        acc_data = acc_data.values

        # Adding in the subject IDs:
        subj_id = np.ones((len(acc_data), 1)) * subj

        # Subj_ID + data
        subj_id_data = np.hstack((subj_id, acc_data))

        all_data = np.vstack((all_data, subj_id_data))

    all_data = all_data[1:, :]
    return all_data


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj,
                                            test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
    print('The train, val and test subjects are: {}')
    print(subjects)

    return subjects


def get_data_from_split(df, args, split, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]

    processed = {'train': {'data': train_data[SENSORS].values,
                           'labels': train_data['gt'].values},
                 'val': {'data': val_data[SENSORS].values,
                         'labels': val_data['gt'].values},
                 'test': {'data': test_data[SENSORS].values,
                          'labels': test_data['gt'].values},
                 'fold': split
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
            len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        print(
            f"{phase}\t data shape = {processed[phase]['data'].shape} \t label shape = {processed[phase]['labels'].shape}")

    # Creating folders by the date now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'all_data', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    save_name = 'motionsense.pkl'

    name = os.path.join(folder, 'unnormalized', save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])
    for phase in ['train', 'val', 'test']:
        processed[phase]['data'] = \
            scaler.transform(processed[phase]['data'])

    # Saving into a joblib file
    name = os.path.join(folder, save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    return processed


def get_data(args):
    acc_folder = 'B_Accelerometer_data'
    args['root_folder'] = args['dataset_loc']

    # Listing all the sub-folders inside both acc_folder and gyro_folder
    acc_path = os.path.join(args['root_folder'], acc_folder)
    print(os.listdir(acc_path))

    # Let us print the number of subjects in each sub-folder
    all_folders = sorted(os.listdir(acc_path))

    _, f_to_a = map_activity_to_id()

    # The dataframe which we will use to store the data into
    cols = {'user': [], 'acc_x': [], 'acc_y': [], 'acc_z': [], 'gt': []}
    df = pd.DataFrame(cols)
    df.head()

    # Starting the loop
    all_data = np.zeros((1, 5))
    for i in tqdm(range(0, len(all_folders))):
        # print('In the folder: {}'.format(all_folders[i]))
        current = all_folders[i]

        # Reading in the data
        data = read_data(current, acc_folder, args)

        # Label
        label = f_to_a[current]
        labels = np.ones((len(data), 1)) * label

        # Stacking them
        both = np.hstack((data, labels))

        all_data = np.vstack((all_data, both))

    # Adding to the data frame
    df['user'] = all_data[1:, 0]
    df['acc_x'] = all_data[1:, 1]
    df['acc_y'] = all_data[1:, 2]
    df['acc_z'] = all_data[1:, 3]
    df['gt'] = all_data[1:, 4]

    print('Done collecting!')

    return df


def prepare_data(args):
    # Loading in all the data first
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    # Performing the train-val-test split
    split = perform_train_val_test_split(unique_subj)
    processed = get_data_from_split(df, args=args, split=split)

    return processed


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = load_args()
    print(args)

    processed = prepare_data(args)
    print('Data preparation complete!')
