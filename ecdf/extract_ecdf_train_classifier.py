import numpy as np
import random

# For classification
from sklearn.ensemble import RandomForestClassifier

# Various metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Sliding window
from ecdf.sliding_window import sliding_window

# Progress tracker
from tqdm.auto import tqdm

# Setting seeds
np.random.seed(42)
random.seed(42)

WINDOW_SIZE = 100  # 2 second of data
OVERLAP = 50  # 50% overlap, as is typical


# From the original DeepConvLSTM implementation
def perform_sliding_window(data_x,
                           data_y,
                           ws,
                           ss):
    """
    Efficiently performing the sliding window based segmentation
    :param data_x: processed data stream
    :param data_y: processed labels stream
    :param ws: window size
    :param ss: overlap size
    :return: windowed data and ground truth labels
    """
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


def generate_windowed_data(processed):
    """
    Generating the windowed data for each of the train-val-test splits
    :param processed: processed data -> to be windowed
    :return: windowed data for all three splits
    """
    segmented_data = {}

    for phase in ['train', 'val', 'test']:
        segmented_data[phase] = {}
        segmented_data[phase]['data'], segmented_data[phase][
            'labels'] = perform_sliding_window(
            processed[phase]['data'],
            processed[phase]['labels'],
            WINDOW_SIZE,
            OVERLAP
        )

        # After segmentation, the data size is:
        print(segmented_data[phase]['data'].shape,
              segmented_data[phase]['labels'].shape)

    return segmented_data


# Taken from: https://github.com/nhammerla/ecdfRepresentation/blob/master
# /python/ecdfRep.py
def ecdfRep(data,
            components):
    #
    #   rep = ecdfRep(data, components)
    #
    #   Estimate ecdf-representation according to
    #     Hammerla, Nils Y., et al. "On preserving statistical
    #     characteristics of
    #     accelerometry data using their empirical cumulative distribution."
    #     ISWC. ACM, 2013.
    #
    #   Input:
    #       data        Nxd     Input data (rows = samples).
    #       components  int     Number of components to extract per axis.
    #
    #   Output:
    #       rep         Mx1     Data representation with M = d*components+d
    #                           elements.
    #
    #   Nils Hammerla '15
    #

    m = np.mean(data, axis=0)

    data = np.sort(data, axis=0)
    data = data[np.int32(
        np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten()

    return np.hstack((data, m))


def compute_ecdf_features(segmented_data):
    """
    Compute the ECDF representations for the segmented windows
    :param segmented_data: segmented data (train-val-test)
    :return: ECDF features with num_components = window size / 2
    """
    # We loop over each window from the train/val/test splits and compute the
    # ECDF features

    ecdf = {}
    num_components = 25

    for phase in tqdm(['train', 'val', 'test']):
        num_windows = segmented_data[phase]['data'].shape[0]
        temp_ecdf = np.zeros((num_windows, (num_components + 1) * 3))

        # Computing the ECDF features for each window
        for i in range(num_windows):
            temp_ecdf[i] = ecdfRep(segmented_data[phase]['data'][i],
                                   components=num_components)

        # Adding to the data dictionary
        ecdf[phase] = temp_ecdf
        print(phase, ecdf[phase].shape)

    return ecdf


def compute_classifier_metrics(actual_labels,
                               pred_labels,
                               phase,
                               running_meter,
                               loss,
                               epoch):
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average='weighted')
    f_score_macro = f1_score(actual_labels, pred_labels, average='macro')
    conf_matrix = confusion_matrix(y_true=actual_labels, y_pred=pred_labels,
                                   normalize="true")
    running_meter.update(phase, loss, acc, f_score_macro, f_score_weighted,
                         conf_matrix, [])

    # printing the metrics
    print("The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
          "f1-score: {:.4f} | weighted f1-score: {:.4f}"
          .format(epoch, phase, loss, acc, f_score_macro, f_score_weighted))

    return running_meter


def train_rf_classifier(ecdf, segmented_data):
    # RF classifier initiation and training
    rf = RandomForestClassifier()
    rf.fit(ecdf['train'], segmented_data['train']['labels'])

    # Predicting on all dataset splits for obtain performance
    log = {}
    for phase in ['train', 'val', 'test']:
        actual_labels = segmented_data[phase]['labels']
        pred_labels = rf.predict(ecdf[phase])

        f_score_macro = f1_score(actual_labels, pred_labels, average='macro')
        print('Phase: {}, mean F1-score: {}'.format(phase, f_score_macro))

        log[phase] = f_score_macro

    return rf, log


def main(processed=None):
    # Obtaining the segmented data first
    segmented_data = generate_windowed_data(processed=processed)

    # Computing the ECDF features
    ecdf = compute_ecdf_features(segmented_data=segmented_data)

    # Training the RF classifier
    train_rf_classifier(ecdf=ecdf, segmented_data=segmented_data)

    return
