import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from solver.losses import *
from solver.utils import split_last_dim

from solver.misc import (
    _handle_unused_kwargs, _select_initial_step, _convert_to_tensor, _scaled_dot_product, _is_iterable,
    _optimal_step_size, _compute_error_ratio, move_to_device, cast_double
)

class RecognitionRNN(tf.keras.Model):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super().__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = tf.keras.layers.Dense(nhidden, activation='tanh')
        self.h2o = tf.keras.layers.Dense(latent_dim * 2)

    #@tf.function
    def call(self, x, h):
        x = cast_double(x)
        h = cast_double(h)

        combined = tf.concat((x, h), axis=1)
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return tf.zeros([self.nbatch, self.nhidden], dtype=tf.float32)

class Decoder(tf.keras.Model):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(obs_dim, activation=None)
        #self.fc2 = tf.keras.layers.Dense(obs_dim)

    def call(self, z):
        z = cast_double(z)

        out = self.fc1(z)
        #out = self.fc2(out)
        return out

class GRU_unit(tf.keras.models.Model):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            #update gate input [latent_dim * 2 + input_dim]
            self.update_gate = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=n_units),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim),
                tf.keras.layers.Activation('sigmoid')])
        else: 
            self.update_gate  = update_gate
        
        if reset_gate is None:
            self.reset_gate =tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=n_units),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim),
                tf.keras.layers.Activation('sigmoid')])
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=n_units),
                tf.keras.layers.Activation('tanh'),
                tf.keras.layers.Dense(latent_dim*2)])
        else: 
            self.new_state_net  = new_state_net
    
    @tf.function
    def call(self, y_mean, y_std, x):
        y_concat = tf.concat([y_mean, y_std, x],-1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = tf.concat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = tf.abs(new_state_std)

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std
        new_y_std = tf.abs(new_y_std)
        return new_y, new_y_std

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
        train_classif_w_reconstr = False,
        **kwargs):
        
        dynamic = kwargs.pop('dynamic', True)
        
        super().__init__(dynamic=dynamic)
        print('dynamic - {}'.format(dynamic))
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
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(300,
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros'),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(z0_dim,
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros')]
        
    def get_gaussian_likelihood(self, truth, pred):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.shape

        # Compute likelihood of the data under the predictions
        truth_repeated = tf.tile(tf.expand_dims(truth,0), [n_traj, 1, 1, 1])

        log_density_data = gaussian_log_density(pred, truth_repeated, self.obsrv_std)
        # shape: [n_traj_samples]
        return tf.reduce_sum(log_density_data, -1)
        
    def get_mse(self, truth, pred):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.shape

        # Compute likelihood of the data under the predictions
        truth_repeated = tf.tile(tf.expand_dims(truth,0), [n_traj, 1, 1, 1])
        log_density_data = mse(pred, truth_repeated)
        # shape: [n_traj_samples]
        return tf.reduce_mean(log_density_data)
        
    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., mode='extrap'):
        # Condition on subsampled points
        # Make predictions for all the points
        # Make sure that time is 1-D vector
        if (len(batch_dict["observed_tp"].shape)==2):
            batch_dict["observed_tp"] = tf.squeeze(batch_dict["observed_tp"])
        if (len(batch_dict["tp_to_predict"].shape)==2):
            batch_dict["tp_to_predict"] = tf.squeeze(batch_dict["tp_to_predict"])
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
                        batch_dict["observed_data"], 
                        batch_dict["observed_tp"], 
                        n_traj_samples = n_traj_samples,
                        mode = mode)
        
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
            
        ce_loss = tf.zeros_like([0.], tf.float32)
        if ("labels" in batch_dict.keys()) and (batch_dict["labels"] is not None) and self.use_binary_classif:

            if (batch_dict["labels"].shape[-1] == 1) or (len(batch_dict["labels"].shape) == 1):
                ce_loss = binary_ce(
                    info["label_predictions"], 
                    batch_dict["labels"])
            else:
                ce_loss = multiclass_ce(
                    info["label_predictions"], 
                    batch_dict["labels"])


        # IWAE Loss
        loss = -tf.reduce_mean(rec_likelihood - kl_coef*kl_div_z0, 0)
        #loss = -tf.reduce_logsumexp(rec_likelihood - kl_coef*kl_div_z0, 0)
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
                n_gru_units = 100,
                backwards_evaluation = True,
                adjoint=True,
                save_info = False,
                **kwargs):
        
        dynamic = kwargs.pop('dynamic', True)
        super().__init__(**kwargs, dynamic=dynamic)
        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim
        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.backwards_evaluation = backwards_evaluation
        self.save_info = save_info
        self.adjoint = adjoint

        if GRU_update is None:
            self.GRU_update = GRU_unit(self.latent_dim, self.input_dim, n_units=n_gru_units)
            #self.GRU_update = tf.keras.layers.GRU(units=self.latent_dim*2, stateful=True, return_sequences=True, bias_initializer='glorot_uniform')
        else:
            self.GRU_update = GRU_update

        #self.var_mean = tf.keras.Sequential([
        #    #tf.keras.layers.InputLayer(input_shape=(n_inputs,)),
        #    tf.keras.layers.Dense(self.latent_dim*2),
        #    tfp.layers.DistributionLambda(
        #        lambda t: tfp.distributions.Normal(loc=t[..., :self.latent_dim],
        #                   scale=1e-2 + tf.math.softplus(0.05 * t[..., self.latent_dim:]))),
        #
        #])

        self.transform_z0 = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100,kernel_initializer='random_normal',
                                             bias_initializer='zeros', activation=None),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dense(self.z0_dim*2, kernel_initializer='random_normal',
                                             bias_initializer='zeros', activation=None)
        ])

    #@tf.function
    def call(self, data, time_steps):
        data, time_steps = tf.cast(data, tf.float32), tf.cast(time_steps, tf.float32)
        n_traj, n_tp, n_dims = data.shape
        backwards = self.backwards_evaluation

        save_info = self.save_info
        if len(time_steps) == 1:
            prev_y = tf.zeros((1, n_traj, self.latent_dim))
            prev_std = tf.zeros((1, n_traj, self.latent_dim))
            x_i = tf.expand_dims(data[:,0,:], 0)
            last_yi, last_std = self.GRU_update(prev_y, prev_std, x_i)
            #y_concat = tf.concat([prev_y, prev_std, x_i])
            #rec_out = self.GRU_update(y_concat)
            #probas = self.var_mean(rec_out)

            #last_yi, last_yi_std = probas.mean(), probas.stddev()
        else:
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = backwards, save_info=save_info)
            means_z0 = tf.reshape(last_yi, [1, n_traj, self.latent_dim])
            std_z0 = tf.reshape(last_yi_std, [1, n_traj, self.latent_dim])

        mean_z0, std_z0 = split_last_dim(self.transform_z0(tf.concat([means_z0, std_z0], -1)))
        std_z0 = tf.abs(std_z0)
        
        if save_info:
            self.extra_info = extra_info
        return mean_z0, std_z0
    
    @tf.function
    def run_odernn(self, data, time_steps, 
        run_backwards = True, save_info=False):

        n_traj, n_tp, n_dims = data.shape

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        prev_y = tf.zeros((1, n_traj, self.latent_dim), tf.float32)
        prev_std = tf.zeros((1, n_traj, self.latent_dim), tf.float32)
        #print(time_steps)
        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        latent_ys = []
        extra_info = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)
        
        def max_step(thresh, prev_t, t_i, min_step):
            return max(2, np.ceil((prev_t - t_i) / minimum_step))

        for t in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = tf.stack((prev_t, t_i))
                #print('case 1 :{}'.format(prev_y))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y)*(t_i-prev_t)

                ode_sol = prev_y + inc
                ode_sol = tf.stack((prev_y, ode_sol), axis=2)
            else:
                #print(prev_t.numpy(), t_i.numpy(), minimum_step.numpy())
                n_intermediate_tp = tf.py_function(max_step, [2, prev_t, t_i, minimum_step], Tout=tf.int32)
               # print('case 2 :'.format(prev_y))
                #n_intermediate_tp = max(2, tf.cast(((prev_t - t_i) / minimum_step),tf.int32))

                time_points = tf.linspace(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points, adjoint=self.adjoint)
            ode_sol = tf.cast(ode_sol, tf.float32)

            if tf.reduce_min(ode_sol[:,:,0,:]-prev_y)>=1e-3:
                print("Error: first point of the ODE is not equal to initial value")
            
            yi_ode = ode_sol[:, :, -1, :]
            x_i = tf.expand_dims(data[:,t,:], 0)
            #y_concat = tf.concat([yi_ode, prev_std, x_i], -1)
            last_yi, last_std = self.GRU_update(yi_ode, prev_std, x_i)
            #probas = self.var_mean(rec_out)
            #last_yi, last_std = probas.mean(), probas.stddev()
            prev_y, prev_std = last_yi, last_std
            prev_t, t_i = time_steps[t], time_steps[t-1]

            latent_ys.append(last_yi)
            if save_info:
                d = {"yi_ode": tf.stop_gradient(yi_ode), 
                     "yi": tf.stop_gradient(last_yi), "yi_std": tf.stop_gradient(last_std), 
                     "time_points": tf.stop_gradient(time_points), "ode_sol": tf.stop_gradient(ode_sol)}
                extra_info.append(d)
        
        return last_yi, last_std, latent_ys, extra_info
