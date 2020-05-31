import os
import logging
import pickle

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import sklearn.metrics as skm
import subprocess
import datetime

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.h5' % epoch)
    tf.keras.models.save_model(model=state, filepath=filename)

def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def inf_generator(iterable):
    """Allows training with tf RepeatDataset in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent

def split_last_dim(data):
    last_dim = data.shape[-1]
    last_dim = last_dim//2
    if len(data.shape) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.shape) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res

def flatten(x, dim):
    return tf.reshape(x, (x.shape[:dim] + (-1, )))

def subsample_timepoints(data, time_steps, n_tp_to_sample = None):
    # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
    if n_tp_to_sample is None:
        return data, time_steps
    n_tp_in_batch = len(time_steps)
    
    if n_tp_to_sample > 1:
        # Subsample exact number of points
        assert(n_tp_to_sample <= n_tp_in_batch)
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.shape[0]):
            missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))
            data[i, missing_idx] = 0.
    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        # Subsample percentage of points from each time series
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.shape[0]):
            n_to_sample = int(n_tp_in_batch * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(time_steps, n_to_sample, replace = False))
            tp_to_set_to_zero = np.setdiff1d(time_steps, subsampled_idx)

            data[i, tp_to_set_to_zero] = 0.

    return data, time_steps

def cut_out_timepoints(data, time_steps, n_points_to_cut = None):
    # n_points_to_cut: number of consecutive time points to cut out
    if n_points_to_cut is None:
        return data, time_steps
    n_tp_in_batch = len(time_steps)
    if n_points_to_cut < 1:
        raise Exception("Number of time points to cut out must be > 1")

    assert(n_points_to_cut <= n_tp_in_batch)
    n_points_to_cut = int(n_points_to_cut)
    for i in range(data.shape[0]):
        start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)
        data[i, start : (start + n_points_to_cut)] = 0.
    return data, time_steps

def split_train_test(data, train_fraq = 0.8):
    n_samples = data.shape[0]
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]
    return data_train, data_test

def split_train_test_data_and_time(data, time_steps, train_fraq = 0.8):
    n_samples = data.shape[0]
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]

    assert(len(time_steps.shape) == 2)
    train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
    test_time_steps = time_steps[:, int(n_samples * train_fraq):]

    return data_train, data_test, train_time_steps, test_time_steps

def get_next_batch(data):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = data.__next__()
    batch_dict = get_dict_template()
    batch_dict = {k:v for k,v in batch_dict.items() if k in data_dict.keys()}
    # remove the time points where there are no observations in this batch
    non_missing_tp = tf.reduce_sum(data_dict["observed_data"],(0,2)) != 0.
    batch_dict["observed_data"] = tf.boolean_mask(data_dict["observed_data"], non_missing_tp, axis=1) 
    batch_dict["observed_tp"] = tf.boolean_mask(tf.squeeze(data_dict["observed_tp"]),non_missing_tp)

    batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] =  tf.squeeze(data_dict["tp_to_predict"])

    non_missing_tp = tf.reduce_sum(data_dict["data_to_predict"],(0,2)) != 0.
    batch_dict["data_to_predict"] = tf.boolean_mask(data_dict["data_to_predict"], non_missing_tp, axis=1)
    batch_dict["tp_to_predict"] =  tf.boolean_mask(tf.squeeze(data_dict["tp_to_predict"]),non_missing_tp)

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    #batch_dict["mode"] = data_dict["mode"]
    return batch_dict

def get_h5_model(h5_path):
    if not os.path.exists(h5_path):
        raise Exception("Checkpoint " + h5_path + " does not exist.")
    # Load checkpoint.
    checkpt = tf.keras.models.load_model(h5_path)
    return checkpt

def update_learning_rate_optimizer(optimizer, initial_lr=0.1, decay_steps=10, lowest = 1e-3, **kwargs):
    # initial_lr: initial learning rate
    # decay_steps: number of steps to perform lr decay until lowest reached
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_lr,
                                                            decay_steps,
                                                            lowest)
    params_dict = {'learning_rate': schedule}
    params_dict.update(kwargs)
    return optimizer(**kwargs)

def reverse(tensor):
    idx = [i for i in range(tensor.shape[0]-1, -1, -1)]
    return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1, 
    n_units = 100, nonlinear = tf.keras.activations.relu):
    model = [#tf.keras.layers.InputLayer(input_shape=n_inputs, ragged=True),
            tf.keras.layers.Dense(units=n_units)]
    for i in range(n_layers):
        #model.add(tf.keras.layers.Activation(nonlinear))
        model.append(tf.keras.layers.Dense(n_units, activation=nonlinear))
    #model.add(tf.keras.layers.Activation(nonlinear))
    model.append(tf.keras.layers.Dense(n_outputs, activation=nonlinear))
    return tf.keras.models.Sequential(model)

def get_item_from_pickle(pickle_file, item_name):
    from_pickle = load_pickle(pickle_file)
    if item_name in from_pickle:
        return from_pickle[item_name]
    return None

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "labels": None
            }

def normalize_data(data):
    reshaped = data.reshape(-1, data.shape[-1])

    att_min = tf.reduce_min(reshaped, 0)[0]
    att_max = tf.reduce_max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[ att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("division by zero")

    if tf.math.is_nan(data_norm).numpy().any():
        raise Exception("data contains nans")

    return data_norm, att_min, att_max

def split_data_extrap(data_dict):
    n_observed_tp = data_dict["data"].shape[1] // 2

    split_dict = {"observed_data": tf.identity(data_dict["data"][:,:n_observed_tp,:]),
                "observed_tp": tf.identity(data_dict["time_steps"][:,:n_observed_tp]),
                "data_to_predict": tf.identity(data_dict["data"][:,n_observed_tp:,:]),
                "tp_to_predict": tf.identity(data_dict["time_steps"][:,n_observed_tp:])}

    if ("labels" in data_dict):
        split_dict["labels"] = tf.identity(data_dict["labels"])

    split_dict["mode"] = "extrap"
    return split_dict

def split_data_interp(data_dict):
    split_dict = {"observed_data": tf.identity(data_dict["data"]),
                "observed_tp": tf.identity(data_dict["time_steps"]),
                "data_to_predict": tf.identity(data_dict["data"]),
                "tp_to_predict": tf.identity(data_dict["time_steps"])}

    if ("labels" in data_dict):
        split_dict["labels"] = tf.identity(data_dict["labels"])

    split_dict["mode"] = "interp"
    return split_dict

def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
    # n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
    # n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points

    if n_tp_to_sample is not None:
        # Randomly subsample time points
        data, time_steps = subsample_timepoints(
            tf.identity(data_dict["observed_data"]), 
            time_steps = tf.identity(data_dict["observed_tp"]), 
            n_tp_to_sample = n_tp_to_sample)

    if n_points_to_cut is not None:
        # Remove consecutive time points
        data, time_steps= cut_out_timepoints(
            tf.identity(data_dict["observed_data"]), 
            time_steps = tf.identity(data_dict["observed_tp"]), 
            n_points_to_cut = n_points_to_cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["observed_data"] = tf.identity(data)
    new_data_dict["observed_tp"] = tf.identity(time_steps)

    if n_points_to_cut is not None:
        # Cut the section in the data to predict as well
        # Used only for the demo on the periodic function
        new_data_dict["data_to_predict"] = tf.identity(data)
        new_data_dict["tp_to_predict"] = tf.identity(time_steps)

    return new_data_dict


def split_and_subsample_batch(data_dict, data_type = "train",
                                extrap=False, sample_tp=None, cut_tp=None):
    if data_type == "train":
        # Training set
        if extrap:
            processed_dict = split_data_extrap(data_dict)
        else:
            processed_dict = split_data_interp(data_dict)   
    else:
        # Test set
        if extrap:
            processed_dict = split_data_extrap(data_dict)
        else:
            processed_dict = split_data_interp(data_dict)
        
    # Subsample points or cut out the whole section of the timeline
    if (sample_tp is not None) or (cut_tp is not None):
        processed_dict = subsample_observed_data(processed_dict, 
            n_tp_to_sample = sample_tp, 
            n_points_to_cut = cut_tp)
    # if (sample_tp is not None):
    #processed_dict = subsample_observed_data(processed_dict, 
    #n_tp_to_sample = sample_tp)
    return processed_dict

def compute_loss_all_batches(model,
    test_dataloader, mode, dataset,
    n_batches, experimentID, device,
    n_traj_samples = 1, kl_coef = 1., 
    max_samples_for_eval = None):

    total = {}
    total["loss"] = 0
    total["likelihood"] = 0
    total["mse"] = 0
    total["kl_first_p"] = 0
    total["std_first_p"] = 0
    total["pois_likelihood"] = 0
    total["ce_loss"] = 0

    n_test_batches = 0

    classif_predictions = tf.zeros_like([])
    all_test_labels = tf.zeros_like([])

    for i in range(n_batches):
        print("Computing loss for batch {}".format(i))

        batch_dict = get_next_batch(test_dataloader)

        results  = model.compute_all_losses(batch_dict,
            n_traj_samples = n_traj_samples, kl_coef = kl_coef)

        if mode=='classification':
            n_labels = model.n_labels #batch_dict["labels"].size(-1)
            n_traj_samples = results["label_predictions"].shape[0]

            classif_predictions = tf.concat([classif_predictions, 
                tf.reshape(results["label_predictions"],[n_traj_samples, -1, n_labels])],1)
            all_test_labels = tf.concat([all_test_labels, 
                tf.reshape(batch_dict["labels"], [-1, n_labels])],0)

            for key in total.keys(): 
                if key in results:
                    var = results[key]
                if isinstance(var, tf.Tensor):
                    var = tf.stop_gradient(var)
                total[key] += var

        n_test_batches += 1

    if n_test_batches > 0:
        for key, value in total.items():
            total[key] = total[key] / n_test_batches
 
    if mode=='classification':
        #all_test_labels = all_test_labels.reshape(-1)
        # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
        all_test_labels = tf.tile(all_test_labels, [n_traj_samples,1,1])


        idx_not_nan = 1 - tf.math.is_nan(all_test_labels)
        classif_predictions = classif_predictions[idx_not_nan]
        all_test_labels = all_test_labels[idx_not_nan]

        dirname = "plots/" + str(experimentID) + "/"
        os.makedirs(dirname, exist_ok=True)

        total["auc"] = 0.
        if tf.reduce_sum(all_test_labels) != 0.:
            print("Number of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
            print("Number of examples with mortality 1: {}".format(tf.reduce_sum(all_test_labels == 1.)))

            # Cannot compute AUC with only 1 class
            total["auc"] = tf.py_function(skm.roc_auc_score, [all_test_labels.numpy().reshape(-1), 
                classif_predictions.numpy().reshape(-1)], Tout=tf.float32)
        else:
            print("Warning: Couldn't compute AUC -- all examples are from the same class")
    return total