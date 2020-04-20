import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
    n_data_points = mu_2d.shape[-1]

    if n_data_points > 0:
        gaussian = tfp.distributions.Independent(tfp.distributions.Normal(loc = mu_2d,
                                                 scale = tf.tile(obsrv_std,n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d) 
        log_prob = log_prob / n_data_points 
    else:
        log_prob = tf.squeeze(tf.zeros([1]))
    return log_prob

def poisson_log_likelihood(log_lambdas, data, indices, int_lambdas):
    n_data_points = data.shape[-1]

    if n_data_points > 0:
        log_prob = tf.reduce_sum(log_lambdas) - int_lambdas[indices]
    else:
        log_prob = tf.squeeze(tf.zeros([1]))
    return log_prob

def mse_wrapper(mu, data, indices=None):
    n_data_points = data.shape[-1]

    if n_data_points > 0:
        loss = tf.keras.losses.MeanSquaredError()(data, mu)
    else:
        loss = tf.squeeze(tf.zeros([1]))
    return loss

def gaussian_log_density(mu, data, obsrv_std):
    if (len(mu.shape) == 3):
        # add additional dimension for gp samples
        mu = tf.expand_dims(mu, 0)

    if (len(data.shape) == 2):
        # add additional dimension for gp samples and time step
        data = tf.expand_dims(tf.expand_dims(data, 0), 2)
    elif (len(data.shape) == 3):
        # add additional dimension for gp samples
        data = tf.expand_dims(data, 0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.shape

    assert(data.shape[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    mu_flat =tf.reshape(mu, [n_traj_samples*n_traj, n_timepoints * n_dims])
    n_traj_samples, n_traj, n_timepoints, n_dims = data.shape
    data_flat = tf.reshape(data, [n_traj_samples*n_traj, n_timepoints * n_dims])

    res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
    res = tf.transpose(tf.reshape(res, [n_traj_samples, n_traj]), perm=[0,1])
    return res

def mse(mu, data):
    if (len(mu.shape) == 3):
        # add additional dimension for gp samples
        mu = tf.expand_dims(mu, 0)

    if (len(data.shape) == 2):
        # add additional dimension for gp samples and time step
        data = tf.expand_dims(tf.expand_dims(data, 0), 2)
    elif (len(data.shape) == 3):
        # add additional dimension for gp samples
        data = tf.expand_dims(data, 0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.shape

    assert(data.shape[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    mu_flat =tf.reshape(mu, [n_traj_samples*n_traj, n_timepoints * n_dims])
    n_traj_samples, n_traj, n_timepoints, n_dims = data.shape
    data_flat = tf.reshape(data, [n_traj_samples*n_traj, n_timepoints * n_dims])

    res = mse_wrapper(mu_flat, data_flat)
    res = tf.transpose(tf.reshape(res, [n_traj_samples, n_traj]), perm=[0,1])
    return res

def poisson_proc_likelihood(truth, pred_y, info):
    # Compute Poisson likelihood
    # Sum log lambdas across all time points

    poisson_log_l = tf.reduce_sum(info["log_lambda_y"], 2) - info["int_lambda"]
    # Sum over data dims
    poisson_log_l = tf.reduce_mean(poisson_log_l, -1)

    return poisson_log_l

def binary_ce(pred_y, truth):

    truth = tf.reshape(truth, (-1,))

    if len(pred_y.shape) == 1:
        pred_y = tf.expand_dims(pred_y, 0)
 
    n_traj_samples = pred_y.shape[0]
    pred_y = pred_y.reshape(n_traj_samples, -1)

    idx_not_nan = 1 - tf.math.is_nan(truth)
    if len(idx_not_nan) == 0:
        print("All are labels are NaNs!")
        ce_loss = tf.zeros_like([])

    pred_y = pred_y[:,idx_not_nan]
    truth = truth[idx_not_nan]

    if tf.reduce_sum(truth == 0.) == 0 or tf.reduce_sum(truth == 1.) == 0:
        print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

    assert(not tf.math.is_nan(pred_y).any())
    assert(not tf.math.is_nan(truth).any())

    # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
    truth = tf.tile(truth.repeat, [n_traj_samples, 1])
    ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(pred_y, truth)

    # divide by number of patients in a batch
    ce_loss = ce_loss / n_traj_samples
    return ce_loss


def multiclass_ce(pred_y, true_label):

    if (len(pred_y.size()) == 3):
        pred_y = tf.expand_dims(pred_y,0)

    n_traj_samples, n_traj, n_tp, n_dims = pred_y.shape

    # For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
    true_label = true_label.repeat(n_traj_samples, 1, 1)

    pred_y = pred_y.reshape(n_traj_samples * n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

    if (pred_y.size(-1) > 1) and (true_label.size(-1) > 1):
        assert(pred_y.size(-1) == true_label.size(-1))
        # targets are in one-hot encoding -- convert to indices
        _, true_label = tf.reduce_max(true_label, -1)

    res = []
    for i in range(true_label.size(0)):
        pred = pred_y[i]
        labels = true_label[i]

        pred = tf.reshape(pred, (-1, n_dims))

        if (len(labels) == 0):
            continue

        ce_loss = tf.keras.losses.CategoricalCrossentropy()(pred, tf.cast(labels, tf.int64))
        res.append(ce_loss)

    ce_loss = tf.stack(res, 0)
    ce_loss = tf.mean(ce_loss)
    return ce_loss
