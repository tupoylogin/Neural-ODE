import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from solver.losses import *

from solver.misc import (
    _handle_unused_kwargs, _select_initial_step, _convert_to_tensor, _scaled_dot_product, _is_iterable,
    _optimal_step_size, _compute_error_ratio, move_to_device, cast_double
)

class RecognitionRNN(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = tf.keras.layers.Dense(nhidden, activation='tanh')
        self.h2o = tf.keras.layers.Dense(latent_dim * 2)

    def call(self, x, h):
        x = cast_double(x)
        h = cast_double(h)

        combined = tf.concat((x, h), axis=1)
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return tf.zeros([self.nbatch, self.nhidden], dtype=tf.float64)


class Decoder(tf.keras.Model):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.fc1 = tf.keras.layers.Dense(nhidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(obs_dim)

    def call(self, z):
        z = cast_double(z)

        out = self.fc1(z)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class VAEBaseline(tf.keras.models.Model):
    """
    Variational Autoencoder wrapper
    """
    def __init__(self, input_dim, latent_dim, 
        z0_prior,
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):
        
        super(VAEBaseline).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels

    	self.obsrv_std =  tf.convert_to_tensor([obsrv_std], dtype=tf.float32)

    	self.z0_prior = z0_prior
    	self.use_binary_classif = use_binary_classif
    	self.classif_per_tp = classif_per_tp
    	self.use_poisson_proc = use_poisson_proc
    	self.linear_classifier = linear_classifier
    	self.train_classif_w_reconstr = train_classif_w_reconstr

        z0_dim = latent_dim
        if use_poisson_proc:
        	z0_dim += latent_dim

        if use_binary_classif: 
            if linear_classifier:
                self.classifier = tf.keras.models.Sequential[
                    tf.keras.layers.Dense(n_labels,
                                             kernel_initializer='random_normal',
                                             bias_initializer='zeros')]
            else: 
                self.classifier = tf.keras.models.Sequential[
                    tf.keras.layers.Dense(n_labels,
                                             kernel_initializer='random_normal',
                                             bias_initializer='zeros'),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(300,
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros'),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(z0_dim,
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros')]
        
        def get_gaussian_likelihood(self, truth, pred):
            # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
            # truth shape  [n_traj, n_tp, n_dim]
            n_traj, n_tp, n_dim = truth.shape

            # Compute likelihood of the data under the predictions
            truth_repeated = tf.tile(truth, [n_traj, 1, 1, 1])

            log_density_data = gaussian_log_density(pred, truth_repeated, self.obsrv_std)
            # shape: [n_traj_samples]
            return tf.reduce_mean(log_density_data, 1)
        
        def get_mse(self, truth, pred):
            # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
            # truth shape  [n_traj, n_tp, n_dim]
            n_traj, n_tp, n_dim = truth.shape

            # Compute likelihood of the data under the predictions
            truth_repeated = tf.tile(truth, [n_traj, 1, 1, 1])
            log_density_data = mse(pred, truth_repeated)
            # shape: [n_traj_samples]
            return tf.reduce_mean(log_density_data)
        
        def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
            # Condition on subsampled points
            # Make predictions for all the points
            pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
                            batch_dict["observed_data"], 
                            batch_dict["observed_tp"], 
                            n_traj_samples = n_traj_samples,
                            mode = batch_dict["mode"])
            
            fp_mu, fp_std, fp_enc = info['first_point']
            fp_distr = tfp.distributions.Normal(loc=fp_mu, scale=fp_std)

            kl_div_z0 = tfp.distributions.kl_divergence(fp_distr, self.z0_prior)
            # Mean over number of latent dimensions
            # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
            # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
            # shape after: [n_traj_samples]
            kl_div_z0 = tf.reduce_mean(kl_div_z0,(1,2))

            # Compute likelihood of all the points
            rec_likelihood = self.get_gaussian_likelihood(
                              batch_dict["data_to_predict"],
                              pred_y)

            mse = self.get_mse(
                   batch_dict["data_to_predict"], 
                   pred_y)

            if self.use_poisson_proc:
                pois_log_likelihood = poisson_proc_likelihood(
                    batch_dict["data_to_predict"], pred_y, 
                    info)
                # Take mean over n_traj
                pois_log_likelihood = tf.reduce_mean(pois_log_likelihood, 1)
            
            ce_loss = tf.zeros_like([0.], tf.float64)
            if (batch_dict["labels"] is not None) and self.use_binary_classif:

                if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                    ce_loss = binary_ce(
                        info["label_predictions"], 
                        batch_dict["labels"])
                else:
                    ce_loss = multiclass_ce(
                        info["label_predictions"], 
                        batch_dict["labels"])


            # IWAE Loss
            loss = -tf.reduce_logsumexp(rec_likelihood - kl_coef*kl_div_z0, 0)
            if tf.math.is_nan(loss):
                loss = -tf.reduce_mean(rec_likelihood - kl_coef*kl_div_z0, 0)

            if self.use_poisson_proc:
                loss -= 0.1*pois_log_likelihood
            
            if self.use_binary_classif:
                if self.train_classif_w_reconstr:
                    loss = loss +  ce_loss * 100
                else:
                    loss =  ce_loss
            
            results = {}
            results["loss"] = tf.reduce_mean(loss)
            results["likelihood"] = tf.stop_gradient(tf.reduce_mean(rec_likelihood))
            results["mse"] = tf.stop_gradient(tf.reduce_mean(mse))
            results["cross_entropy"] = tf.stop_gradient(tf.reduce_mean(ce_loss))
            results["kl_first_p"] =  tf.stop_gradient(tf.reduce_mean(kl_div_z0))
            results["std_first_p"] = tf.stop_gradient(tf.reduce_mean(fp_std))

            if self.use_poisson_proc:
                results["pois_likelihood"] = tf.stop_gradient(tf.reduce_mean(pois_log_likelihood))
            
            return results

class ODERNNEncoder(tf.keras.models.Model):
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
                z0_dim = None,
                GRU_update=None,
                n_gru_units = 100):
        super(ODERNNEncoder).__init__()
        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim
        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        if GRU_update is None:
            self.GRU_update = tf.keras.layers.GRU(units=n_gru_units, return_sequences=True)
        else:
            self.GRU_update = GRU_update

        self.var_mean = tf.keras.Sequential([
            tf.keras.layers.Dense(2),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(loc=t[..., :1],
                           scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),

        ])

        self.transform_z0 = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100,kernel_initializer='random_normal',
                                             bias_initializer='zeros', activation=None),
            tf.keras.layers.Tanh(),
            tfp.layers.Dense(self.z0_dim, kernel_initializer='random_normal',
                                             bias_initializer='zeros', activation=None)
        ])

    @func_cast_double
    def call(self, data, truth, time_steps, backwards=True, save_info = False):
        n_traj, n_tp, n_dims = data.shape
        if len(time_steps) == 1:
            prev_y = tf.zeros((1, n_traj, self.latent_dim))
            prev_std = tf.zeros((1, n_traj, self.latent_dim))
            x_i = tf.expand_dims(data[:,0,:], 0)
            y_concat = tf.concat([prev_y, prev_std, x_i])
            rec_out = self.GRU_update(y_concat)
            probas = self.var_mean(rec_out)

            last_yi, last_std = probas.mean(), probas.stddev()
        else:
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = backwards, save_info=save_info)
            means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
            std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)
        
        if save_info:
            self.extra_info = extra_info
        return means_z0, std_z0

    def run_odernn(self, data, time_steps, 
        run_backwards = True, save_info=False):

        n_traj, n_tp, n_dims = data.shape

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        prev_y = tf.zeros((1, n_traj, self.latent_dim))
        prev_std = tf.zeros((1, n_traj, self.latent_dim))

        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        latent_ys = []
        extra_info = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)
        
        for t in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = tf.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y)*(t_i-prev_t)

                ode_sol = prev_y + inc
                ode_sol = tf.stack((prev_y, ode_sol), axis=2)
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = tf.linspace(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)
            
            if tf.reduce_min(ode_sol[:,:,0,:]-prev_y)>=1e-3:
                print("Error: first point of the ODE is not equal to initial value")
            
            yi_ode = ode_sol[:, :, -1, :]
            x_i = tf.expand_dims(data[:,t,:], 0)
            y_concat = tf.concat([yi_ode, prev_std, x_i])
            rec_out = self.GRU_update(y_concat)
            probas = self.var_mean(rec_out)
            last_yi, last_std = probas.mean(), probas.stddev()
            prev_y, prev_std = last_yi, last_std
            prev_t, t_i = time_steps[t], time_steps[t-1]

            latent_ys.append(last_yi)
            if save_info:
                d = {"yi_ode": yi_ode.detach(), 
                     "yi": tf.stop_gradient(last_yi), "yi_std": tf.stop_gradient(last_std), 
                     "time_points": tf.stop_gradient(time_points), "ode_sol": tf.stop_gradient(ode_sol)}
                extra_info.append(d)
        
        return last_yi, last_std, latent_ys, extra_info
