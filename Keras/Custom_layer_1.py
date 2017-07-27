from keras.layers import Input
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import Model

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# define a self layer which doesn't include trainable parameters
class MyLayer(Layer):

    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self, axis, **kwargs):
        self.axis = axis
        self.result = None
        super(MyLayer, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        print(input_shape)
        super(MyLayer, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, inputs, **kwargs):
        a = inputs[0]
        b = inputs[1]
        self.result = K.concatenate([a, b], axis=self.axis)
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


var_1 = Input(shape=(1,))
var_2 = Input(shape=(2,))

var_test = MyLayer(axis=1)([var_1, var_2])

model = Model(inputs=[var_1, var_2], outputs=[var_test])
model.compile(optimizer='adam', loss='mse')

print(model.summary())
