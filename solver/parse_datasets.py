import os

import numpy as np
import pandas as pd
import tensorflow as tf

from solver.utils import *

def csv_dataset(path=None, type='extrap', dataset=None, time_steps=None, batch_size=64, tp_to_extrap=64, tp_to_interp=100):

    extrap_ = type=='extrap'

    def get_data_dict(data, timesteps, data_type='train'):
        # batch = tf.stack(batch)
        data_dict = {'data': data,
                    'time_steps':timesteps}

        data_dict = split_and_subsample_batch(data_dict,data_type, extrap=extrap_)
        mode = data_dict['mode']
        data_dict = {k:v for k,v in data_dict.items() if k!='mode'}
        return data_dict, mode

    batch_size = batch_size
    time_points_to_extrap = tp_to_extrap

    if path is not None:    
        if os.path.exists(path):
            dataset = pd.read_csv(path)
            dataset = dataset.values
        else:
            raise FileNotFoundError()
    elif dataset is not None:
        dataset = dataset
    else:
        raise AttributeError('Either path or datset must be specified')

    if (len(dataset.shape)==2):
        #we need to duplicate dataset in order to evaluate dynamics via test loop
        dataset = np.tile(np.expand_dims(dataset, axis=0),[2,1,1])
    n_tp = dataset.shape[1]
    n_traj = dataset.shape[0]

    if time_steps is None:
        time_steps = np.linspace(0,1,n_tp).astype('float32')
        #time_steps = time_steps / len(time_steps)
    else:
        time_steps = time_steps.astype('float32')


    if type=='interp':
        # Creating dataset for interpolation
        n_reduced_tp = tp_to_interp
        # sample time points from different parts of the timeline, 
        # so that the model learns from different parts of trajectory
        start_ind = np.random.randint(0, high=n_tp - n_reduced_tp +1, size=n_traj)
        end_ind = start_ind + n_reduced_tp
        sliced = []
        for i in range(n_traj):
            sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
            dataset = sliced
        time_steps = time_steps[:n_reduced_tp]
    #dataset = tf.convert_to_tensor(dataset, tf.float32)
    #time_steps = tf.convert_to_tensor(time_steps)
    time_steps = np.expand_dims(time_steps, 0)
    train_y, test_y = split_train_test(dataset, train_fraq = 0.8)

    n_samples = len(dataset)
    input_dim = dataset.shape[-1]

    train_data, _ = get_data_dict(train_y, time_steps, data_type='train')
    test_data, mode = get_data_dict(test_y, time_steps, data_type='test')
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    data_objects = {"train_dataloader": inf_generator(train_data.cache().batch(batch_size).repeat()), 
                "test_dataloader": inf_generator(test_data.batch(batch_size).repeat()),
                "input_dim": input_dim,
                'mode': mode,
                "n_train_batches": int(np.ceil(train_y.shape[1]/batch_size)),
                "n_test_batches": int(np.ceil(test_y.shape[1]/batch_size))}

    return data_objects

    