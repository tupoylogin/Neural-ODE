import keras
from keras import backend as K
import tensorflow as tf
from keras.layers import Layer

class ODEBlock2D(Layer):
    
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters= filters
        self.kernel_size=kernel_size
        super(ODEBlock2D,self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size+(self.filters+1, self.filters), initializer='lecun_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size+(self.filters+1, self.filters), initializer='lecun_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        super(ODEBlock2D,self).build(input_shape)
    
    def call(self,x):
        t = K.constant([0,1], dtype='float32')
        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv2d(y, self.conv2d_w1, padding='same')
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = self.concat_t(y, t)
        y = K.conv2d(y, self.conv2d_w2, padding='same')
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat([
            tf.shape(x)[:-1],
            tf.constant([1], dtype='int32', shape=(1,))
        ], axis=0)

        t = tf.ones(shape=new_shape)*tf.reshape(t, (1,1,1,1))

        return tf.concat([x, t], axis=-1)

class ODEBlock1D(Layer):
    
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters=filters
        self.kernel_size=kernel_size
        super(ODEBlock1D,self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv1d_w1 = self.add_weight("conv1d_w1", self.kernel_size+self.filters, initializer='lecun_uniform')
        self.conv2d_b1 = self.add_weight("conv1d_b1", (self.filters,), initializer='zero')
        super(ODEBlock1D,self).build(input_shape)
    
    def call(self,x):
        t = K.constant([0,1], dtype='float32')
        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv1d(y, self.conv1d_w1, padding='same')
        y = K.bias_add(y, self.conv1d_b1)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat([
            tf.shape(x)[:-1],
            tf.constant([1], dtype='int32', shape=(1,))
        ], axis=0)

        t = tf.ones(shape=new_shape)*tf.reshape(t, (1,1,1,1))

        return tf.concat([x, t], axis=-1)

class ODEWrapper():
    
    @staticmethod
    def concat_t(x, t):
        new_shape = tf.concat([
            tf.shape(x)[:-1],
            tf.constant([1], dtype='int32', shape=(1,))
        ], axis=0)

        t = tf.ones(shape=new_shape)*tf.reshape(t, (1,1,1,1))

        return tf.concat([x, t], axis=-1)
    
    @staticmethod
    def integrate(func, x):
        t = K.constant([0,1], dtype='float32')
        return tf.contrib.integrate.odeint(func, x, t, rtol=1e-3, atol=1e-3)[1]

