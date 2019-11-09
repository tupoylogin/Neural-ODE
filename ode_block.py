import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer
from solver.odeint import odeint
from solver.adjoint import odeint_adjoint


MAX_NUM_STEPS = 1000


class ODEFunc(Model):

    def __init__(self, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu',
                 **kwargs):
        """
        MLP modeling the derivative of ODE system.
        # Arguments:
            input_dim : int
                Dimension of data.
            hidden_dim : int
                Dimension of hidden layers.
            augment_dim: int
                Dimension of augmentation. If 0 does not augment ODE,
                otherwise augments
                it with augment_dim dimensions.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """

        dynamic = kwargs.pop('dynamic', True)
        super(ODEFunc, self).__init__(**kwargs, dynamic=dynamic)
        self.augment_dim = augment_dim
        # self.data_dim = input_dim
        # self.input_dim = input_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        self.fc1 = keras.layers.Dense(hidden_dim)
        self.fc2 = keras.layers.Dense(hidden_dim)

        self.fc3 = None

        if non_linearity == 'relu':
            self.non_linearity = keras.layers.ReLU()
        elif non_linearity == 'softplus':
            self.non_linearity = keras.layers.Activation('softplus')
        else:
            self.non_linearity = keras.layers.Activation(non_linearity)

    def build(self, input_shape):
        if len(input_shape) > 0:
            self.fc3 = keras.layers.Dense(input_shape[-1])
            self.built = True

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Forward pass. If time dependent, concatenates the time
        dimension onto the input before the call to the dense layer.
        # Arguments:
            t: Tensor. Current time. Shape (1,).
            x: Tensor. Shape (batch_size, input_dim).
        # Returns:
            Output tensor of forward pass.
        """

        # build the final layer if it wasnt built yet
        if self.fc3 is None:
            self.fc3 = keras.layers.Dense(x.shape.as_list()[-1])

        # Forward pass of model corresponds to one function evaluation, so
        # increment counter

        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = tf.ones([x.shape[0], 1], dtype=t.dtype) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = tf.concat([t_vec, x], axis=-1)
            # Shape (batch_size, hidden_dim)
            # TODO: Remove cast when Keras supports double
            out = self.fc1(tf.cast(t_and_x, tf.float32))
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


class ODEBlock(Model):

    def __init__(self, odefunc, is_conv=False, tol=1e-3, adjoint=False,
                 solver='dopri5', **kwargs):
        """
        Solves ODE defined by odefunc.
        # Arguments:
            odefunc : ODEFunc instance or Conv2dODEFunc instance
                Function defining dynamics of system.
            is_conv : bool
                If True, treats odefunc as a convolutional model.
            tol : float
                Error tolerance.
            adjoint : bool
                If True calculates gradient with adjoint solver, otherwise
                backpropagates directly through operations of ODE solver.
            solver: ODE solver. Defaults to DOPRI5.
        """
        dynamic = kwargs.pop('dynamic', True)
        super(ODEBlock, self).__init__(**kwargs, dynamic=dynamic)

        self.adjoint = adjoint
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol
        self.method = solver
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if solver == "dopri5":
            self.options = {'max_num_steps': MAX_NUM_STEPS}
        else:
            self.options = None

    def call(self, x, training=None, eval_times=None, **kwargs):
        """
        Solves ODE starting from x.
        # Arguments:
            x: Tensor. Shape (batch_size, self.odefunc.data_dim)
            eval_times: None or tf.Tensor.
                If None, returns solution of ODE at final time t=1. If tf.Tensor
                then returns full ODE trajectory evaluated at points in eval_times.
        # Returns:
            Output tensor of forward pass.
        """

        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter

        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = tf.convert_to_tensor([0, 1], dtype=x.dtype)
        else:
            integration_time = tf.cast(eval_times, x.dtype)

        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                if self.channel_axis == 1:
                    batch_size, _, height, width = x.shape

                    aug = tf.zeros([batch_size, self.odefunc.augment_dim,
                                    height, width], dtype=x.dtype)

                else:
                    batch_size, height, width, _ = x.shape

                    aug = tf.zeros([batch_size, height, width,
                                    self.odefunc.augment_dim], dtype=x.dtype)

                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = tf.concat([x, aug], axis=self.channel_axis)
            else:
                # Add augmentation
                aug = tf.zeros([x.shape[0], self.odefunc.augment_dim], dtype=x.dtype)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = tf.concat([x, aug], axis=-1)
        else:
            x_aug = x

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method=self.method,
                                 options=self.options)
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.method,
                         options=self.options)

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = tf.linspace(0., 1., timesteps)
        return self.call(x, eval_times=integration_time)


class ODENet(Model):

    def __init__(self, hidden_dim, output_dim,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):
        """
        An ODEBlock followed by a Linear layer.
        # Arguments:
            hidden_dim : int
                Dimension of hidden layers.
            output_dim : int
                Dimension of output after hidden layer. Should be 1 for regression or
                num_classes for classification.
            augment_dim: int
                Dimension of augmentation. If 0 does not augment ODE, otherwise augments
                it with augment_dim dimensions.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
            tol : float
                Error tolerance.
            adjoint : bool
                If True calculates gradient with adjoint method, otherwise
                backpropagates directly through operations of ODE solver.
            solver: ODE solver. Defaults to DOPRI5.
        """
        dynamic = kwargs.pop('dynamic', True)
        super(ODENet, self).__init__(**kwargs, dynamic=dynamic)

        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ODEFunc(hidden_dim, augment_dim,
                          time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, tol=tol, adjoint=adjoint, solver=solver)
        self.linear_layer = keras.layers.Dense(self.output_dim)

    # @tf.function
    def call(self, x, training=None, return_features=False):
        features = self.odeblock(x, training=training)

        # Remove cast when keras supports double
        pred = self.linear_layer(tf.cast(features, tf.float32))
        if return_features:
            return features, pred
        return pred


class Conv2dTime(Model):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, dim_out, kernel_size=3, stride=1, padding="valid", dilation=1,
                 bias=True, transpose=False):
        super(Conv2dTime, self).__init__()
        module = keras.layers.Conv2DTranspose if transpose else tf.keras.layers.Conv2D

        self._padding = padding
        self._layer = module(
            dim_out, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=self._padding,
            dilation_rate=dilation,
            use_bias=bias
        )

        self.channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        # Remove cast when Keras supports double
        t = tf.cast(t, x.dtype)

        if self.channel_axis == 1:
            # Shape (batch_size, 1, height, width)
            tt = tf.ones_like(x[:, :1, :, :], dtype=t.dtype) * t  # channel dim = 1

        else:
            # Shape (batch_size, height, width, 1)
            tt = tf.ones_like(x[:, :, :, :1], dtype=t.dtype) * t  # channel dim = -1

        ttx = tf.concat([tt, x], axis=self.channel_axis)  # concat at channel dim

        # Remove cast when Keras supports double
        ttx = tf.cast(ttx, tf.float32)
        return self._layer(ttx)


class Conv2dODEFunc(Model):

    def __init__(self, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu', **kwargs):
        """
        Convolutional block modeling the derivative of ODE system.
        # Arguments:
            num_filters : int
                Number of convolutional filters.
            augment_dim: int
                Number of augmentation channels to add. If 0 does not augment ODE.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODEFunc, self).__init__(**kwargs, dynamic=dynamic)

        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        # self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = None

        else:
            self.conv1 = tf.keras.layers.Conv2D(self.num_filters,
                                                kernel_size=(1, 1), strides=(1, 1),
                                                padding='valid')
            self.conv2 = tf.keras.layers.Conv2D(self.num_filters,
                                                kernel_size=(3, 3), strides=(1, 1),
                                                padding='same')
            self.conv3 = None

        if non_linearity == 'relu':
            self.non_linearity = tf.keras.layers.ReLU()
        elif non_linearity == 'softplus':
            self.non_linearity = tf.keras.layers.Activation('softplus')
        else:
            self.non_linearity = tf.keras.layers.Activation(non_linearity)

    def build(self, input_shape):
        if len(input_shape) > 0:
            if self.time_dependent:
                self.conv3 = Conv2dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0)
            else:
                self.conv3 = tf.keras.layers.Conv2D(self.channels,
                                                    kernel_size=(1, 1), strides=(1, 1),
                                                    padding='valid')

            self.built = True

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Parameters
        ----------
        t : Tensor
            Current time.
        x : Tensor
            Shape (batch_size, input_dim)
        """
        # build the final layer if it wasnt built yet
        if self.conv3 is None:
            channel_dim = 1 if K.image_data_format() == 'channel_first' else -1
            self.channels = x.shape.as_list()[channel_dim]

            if self.time_dependent:
                self.conv3 = Conv2dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0)
            else:
                self.conv3 = keras.layers.Conv2D(self.channels,
                                                 kernel_size=(1, 1), strides=(1, 1),
                                                 padding='valid')

        self.nfe += 1

        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            # TODO: Remove cast to tf.float32 once Keras supports tf.float64
            x = tf.cast(x, tf.float32)
            out = self.conv1(x)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
            out = self.conv3(out)

        return out


class Conv1dTime(Model):
    """
    Implements time dependent 1d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, dim_out, kernel_size=3, stride=1, padding="valid", dilation=1,
                 bias=True, data_format="channels_last"):
        super(Conv1dTime, self).__init__()
        module = keras.layers.Conv1D

        self.data_format = data_format
        self._padding = padding
        self._layer = module(
            dim_out, kernel_size=kernel_size, strides=stride, padding=self._padding,
            dilation_rate=dilation,
            use_bias=bias, data_format=self.data_format
        )

        self.channel_axis = 1 if self.data_format == 'channels_first' else -1

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        # Remove cast when Keras supports double
        t = tf.cast(t, x.dtype)

        if self.channel_axis == 1:
            # Shape (batch_size, 1, length)
            tt = tf.ones_like(x[:, :1, :], dtype=t.dtype) * t  # channel dim = 1

        else:
            # Shape (batch_size, length, 1)
            tt = tf.ones_like(x[:, :, :1], dtype=t.dtype) * t  # channel dim = -1

        ttx = tf.concat([tt, x], axis=self.channel_axis)  # concat at channel dim

        # Remove cast when Keras supports double
        ttx = tf.cast(ttx, tf.float32)
        return self._layer(ttx)


class Conv1dODEFunc(Model):

    def __init__(self, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu', data_format="channels_last", **kwargs):
        """
        Convolutional block modeling the derivative of ODE system.
        # Arguments:
            num_filters : int
                Number of convolutional filters.
            augment_dim: int
                Number of augmentation channels to add. If 0 does not augment ODE.
            time_dependent : bool
                If True adds time as input, making ODE time dependent.
            non_linearity : string
                One of 'relu' and 'softplus'
        """
        dynamic = kwargs.pop('dynamic', True)
        super(Conv1dODEFunc, self).__init__(**kwargs, dynamic=dynamic)

        self.augment_dim = augment_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        # self.channels += augment_dim
        self.num_filters = num_filters
        self.data_format

        if time_dependent:
            self.conv1 = Conv1dTime(self.num_filters,
                                    kernel_size=1, stride=1, padding=0, data_format=self.data_format)
            self.conv2 = Conv1dTime(self.num_filters,
                                    kernel_size=3, stride=1, padding=1, data_format=self.data_format)
            self.conv3 = None

        else:
            self.conv1 = keras.layers.Conv1D(self.num_filters,
                                             kernel_size=1, strides=1,
                                             padding='valid', data_format=self.data_format)
            self.conv2 = keras.layers.Conv1D(self.num_filters,
                                             kernel_size=3, strides=1,
                                             padding='same', data_format=self.data_format)
            self.conv3 = None

        if non_linearity == 'relu':
            self.non_linearity = keras.layers.ReLU()
        elif non_linearity == 'softplus':
            self.non_linearity = keras.layers.Activation('softplus')
        else:
            self.non_linearity = keras.layers.Activation(non_linearity)

    def build(self, input_shape):
        if len(input_shape) > 0:
            if self.time_dependent:
                self.conv3 = Conv1dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0, data_format=self.data_format)
            else:
                self.conv3 = keras.layers.Conv1D(self.channels,
                                                 kernel_size=1, strides=1,
                                                 padding='valid', data_format=self.data_format)

            self.built = True

    @tf.function
    def call(self, t, x, training=None, **kwargs):
        """
        Parameters
        ----------
        t : Tensor
            Current time.
        x : Tensor
            Shape (batch_size, input_dim)
        """
        # build the final layer if it wasnt built yet
        if self.conv3 is None:
            channel_dim = 1 if self.data_format == 'channel_first' else -1
            self.channels = x.shape.as_list()[channel_dim]

            if self.time_dependent:
                self.conv3 = Conv1dTime(self.channels,
                                        kernel_size=1, stride=1, padding=0)
            else:
                self.conv3 = keras.layers.Conv1D(self.channels,
                                                 kernel_size=(1, 1), strides=(1, 1),
                                                 padding='valid')

        self.nfe += 1

        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            # TODO: Remove cast to tf.float32 once Keras supports tf.float64
            x = tf.cast(x, tf.float32)
            out = self.conv1(x)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
            out = self.conv3(out)

        return out


class Conv2dODENet(Model):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.
    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (channels, height, width).
    num_filters : int
        Number of convolutional filters.
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    return_sequences : bool
        Whether to return the Convolution outputs, or the features after an
        affine transform.
    solver: ODE solver. Defaults to DOPRI5.
    """
    def __init__(self, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, out_kernel_size=(1, 1),
                 non_linearity='relu', out_strides=(1, 1),
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):

        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODENet, self).__init__(**kwargs, dynamic=dynamic)

        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.solver = solver
        self.output_kernel = out_kernel_size
        self.output_strides = out_strides

        odefunc = Conv2dODEFunc(num_filters, augment_dim,
                                time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint, solver=solver)

        self.output_layer = keras.layers.Conv2D(self.output_dim,
                                                kernel_size=out_kernel_size,
                                                strides=out_strides,
                                                padding='same')

    def call(self, x, training=None, return_features=False):
        features = self.odeblock(x, training=training)

        # TODO: Remove cast when Keras supports double
        pred = self.output_layer(tf.cast(features, tf.float32))

        if return_features:
            return features, pred
        else:
            return pred


class Conv1dODENet(Model):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.
    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (sequence, channels, length).
    num_filters : int
        Number of convolutional filters.
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    return_sequences : bool
        Whether to return the Convolution outputs, or the features after an
        affine transform.
    solver: ODE solver. Defaults to DOPRI5.
    """
    def __init__(self, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, out_kernel_size=1,
                 non_linearity='relu', out_strides=1,
                 tol=1e-3, adjoint=False, solver='dopri5', **kwargs):

        dynamic = kwargs.pop('dynamic', True)
        super(Conv2dODENet, self).__init__(**kwargs, dynamic=dynamic)

        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.solver = solver
        self.output_kernel = out_kernel_size
        self.output_strides = out_strides

        odefunc = Conv1dODEFunc(num_filters, augment_dim,
                                time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint, solver=solver)

        self.output_layer = keras.layers.Conv1D(self.output_dim,
                                                kernel_size=out_kernel_size,
                                                strides=out_strides,
                                                padding='same')

    def call(self, x, training=None, return_features=False):
        features = self.odeblock(x, training=training)

        # TODO: Remove cast when Keras supports double
        pred = self.output_layer(tf.cast(features, tf.float32))

        if return_features:
            return features, pred
        else:
            return pred
