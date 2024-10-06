import numpy as np
import os
import pickle
import random
import torch
from datetime import date
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_best_metrics(running_meter, best_meter, classifier=False):
    """
    To compute the best validation loss from the running meter object
    :param running_meter: running meter object with all values
    :param best_meter: updating the best meter based on current running meter
    :return: best validation f1-score
    """
    if classifier:
        loc = np.argmax(running_meter.f1_score['val'])
    else:
        min_loss = np.min(running_meter.loss['val'])  # Minimum loss
        loc = np.where(running_meter.loss['val'] == min_loss)[
            0][-1]  # The latest epoch to give the lowest loss

    # Epoch where the best validation loss was obtained
    epoch = running_meter.epochs[loc]

    # Updating the best meter with values based on the epoch
    phases = ['train', 'val', 'test'] if classifier else ['train', 'val']
    for phase in phases:
        best_meter.update(
            phase,
            running_meter.loss[phase][loc],
            running_meter.accuracy[phase][loc],
            running_meter.f1_score[phase][loc],
            running_meter.f1_score_weighted[phase][loc],
            running_meter.confusion_matrix[phase][loc],
            epoch)

    return best_meter


def compute_metrics(actual_labels, pred_labels, phase, running_meter, loss,
                    epoch):
    """
    Computing the metrics from data
    :param actual_labels: ground truth labels
    :param pred_labels: predicted labels
    :param phase: train/val/test phases
    :param running_meter: logs
    :param loss: loss for the phase
    :param epoch: current epoch
    :return: --
    """
    acc = accuracy_score(actual_labels, pred_labels)
    f_score_weighted = f1_score(actual_labels, pred_labels, average='weighted')
    f_score_macro = f1_score(actual_labels, pred_labels, average='macro')
    conf_matrix = confusion_matrix(y_true=actual_labels, y_pred=pred_labels,
                                   normalize="true")
    running_meter.update(phase, loss, acc, f_score_macro, f_score_weighted,
                         conf_matrix)

    # printing the metrics
    # print("The epoch: {} | phase: {} | loss: {:.4f} | accuracy: {:.4f} | mean "
    #       "f1-score: {:.4f} | weighted f1-score: {:.4f}"
    #       .format(epoch, phase, loss, acc, f_score_macro, f_score_weighted))

    return


def update_loss(phase, running_meter, loss, epoch):
    """
    Updating the loss with the logging meter
    :param phase: train/val/test phases
    :param running_meter: logger
    :param loss: loss value
    :param epoch: current epoch
    :return: --
    """
    running_meter.update(phase, loss, 0, 0, 0, [])

    # printing the metrics
    # print("The epoch: {} | phase: {} | loss: {:.4f}"
    #       .format(epoch, phase, loss))

    return


def model_save_name(args, classifier=False):
    """
    Name to save the model/logs with, based on arguments
    :param args: arg parser
    :param classifier: boolean flag for classifier or not
    :param capture: boolean flag for whether Capture-24 is dataset or not
    :return: name for saving the model/log
    """

    simclr = "simclr_{}_{}_wd_{}_{}".format(
        args['dataset'], args['learning_rate'],
        args['weight_decay'], args['batch_size'])

    # Classifier
    classification = ""
    if classifier:
        if args['saved_model'] is not None:  # i.e., we are using learned
            # weights
            classification += '_saved_model_True'

        classification += "_cls_lr_{}_cls_wd_{}_{}_cls_bs_{}_{}".format(
            args['classifier_lr'], args['classifier_wd'],
            args['learning_schedule'], args['classifier_batch_size'],
            args['classification_model']
        )

    # Random seed
    random_seed = "_rs_{}".format(args['random_seed'])

    name = simclr + classification + random_seed

    return name


def save_meter(args, running_meter, classifier=False):
    """
    Saving the logs
    :param args: arguments
    :param running_meter: running meter object to save
    :param mlp: if saving during the MLP training, then adds '_eval_log.pkl'
    to the end
    :return: nothing
    """
    name = model_save_name(args, classifier=classifier)
    save_name = name + '_eval_log.pkl' if classifier else name + '_log.pkl'

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'saved_logs', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, save_name), 'wb') as f:
        pickle.dump(running_meter, f, pickle.HIGHEST_PROTOCOL)

    return


def save_model(model, args, epoch):
    """
    Saves the weights from the model
    :param model: model being trained
    :param args: arguments
    :return: --
    """
    name = model_save_name(args)

    # Creating logs by the dat now. To make stuff easier
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(dir_path, 'saved_weights', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    model_name = os.path.join(folder, name + '.pkl')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, model_name)

    return


def set_all_seeds(random_seed):
    """
    Setting all seeds during training
    :param random_seed: random seed to set
    :return: --
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    return
