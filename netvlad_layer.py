import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D

# =============================================================================
#
# This script contains the NetVLAD layer for Tensorflow 2.
# outdim does not seem necessary, but was kept to ensure backwards compatibility during testing
#
# =============================================================================

class NetVLAD(layers.Layer):
    """Creates a NetVLAD class.
    """
    def __init__(self, K=64, assign_weight_initializer=None, 
            cluster_initializer=None, skip_postnorm=False, outdim=32768, **kwargs):
        
        self.K = K
        self.assign_weight_initializer = assign_weight_initializer
        self.skip_postnorm = skip_postnorm
        self.outdim = outdim
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.D = input_shape[-1]
        self.C = self.add_weight(name='cluster_centers',
                                 shape=(1,1,1,self.D,self.K),
                                 initializer='zeros',
                                 dtype='float32',
                                 trainable=True)

        self.conv = Conv2D(filters = self.K, kernel_size=1, strides = (1,1),
                           use_bias=True, padding = 'valid',
                           kernel_initializer='zeros')
        self.conv.build(input_shape) 

        super(NetVLAD, self).build(input_shape)

    def call(self, inputs):
        
        s = self.conv(inputs)
        a = tf.nn.softmax(s)

        a = tf.expand_dims(a,-2)

        # VLAD core.
        v = tf.expand_dims(inputs,-1)+self.C
        v = a*v
        v = tf.reduce_sum(v,axis=[1,2])
        v = tf.transpose(v,perm=[0,2,1])

        if not self.skip_postnorm:
            # Result seems to be very sensitive to the normalization method
            # details, so sticking to matconvnet-style normalization here.
            v = self.matconvnetNormalize(v, 1e-12)
            v = tf.transpose(v, perm=[0, 2, 1])
            v = self.matconvnetNormalize(layers.Flatten()(v), 1e-12) #- https://stackoverflow.com/questions/53153790/tensor-object-has-no-attribute-lower

        return v

    def matconvnetNormalize(self, inputs, epsilon):
        return inputs / tf.sqrt(tf.reduce_sum(inputs ** 2, axis=-1, keepdims=True)
                                + epsilon)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.outdim])
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'K': self.K,
            'assign_weight_initializer': self.assign_weight_initializer,
            'skip_postnorm': self.skip_postnorm,
            'outdim': self.outdim
        })
        return config