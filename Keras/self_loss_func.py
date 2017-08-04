from __future__ import print_function, division
from keras.layers import Dense, Input, Embedding, Dropout, GRU
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional

from keras.optimizers import RMSprop
from keras import metrics
from keras import backend as K
from keras.losses import mse
import tensorflow as tf
import numpy as np
import random

# define self sigmoid cross-entroy loss function
def my_loss(y_pred, y_true):
    y_true = K.sigmoid(y_true)
    y_pred = K.sigmoid(y_pred)
    neg_true = 1. - y_true
    neg_pred = 1. - y_pred
    return tf.reduce_sum(y_pred * tf.log(y_true) + neg_pred * tf.log(neg_true),
                         axis=len(y_pred.get_shape()) - 1)

feature_len = 2
units = 8
pre_len = 2
embedding_size = 3
feature_size = 4

x_feature = np.array([[1, 2], [3, 4]])
x_prelen = np.array([[2, 3], [4, 3]])
y = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

embedding_layer = Embedding(output_dim=embedding_size,
                            input_dim=feature_size + 1,
                            input_length=pre_len)

data = Input(shape=(pre_len,), name='scan_input')
x = embedding_layer(data)

feature_input = Input(shape=(feature_len,), name='feature_input')
temp_out = embedding_layer(feature_input)
feature_output = TimeDistributed(Dense(units))(temp_out)
initial_state = K.mean(feature_output, axis=1)

lstm_output = GRU(units=units, return_sequences=False)(x, initial_state=initial_state)

lstm_output = Dropout(0.1)(lstm_output)
output = Dense(feature_size, activation='softmax', name='scan_output')(lstm_output)

model = Model(inputs=[feature_input, data],
              outputs=[output])

optimizer = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-6, decay=0)
model.compile(optimizer=optimizer,
              loss=my_loss,
              metrics=[metrics.categorical_accuracy, metrics.top_k_categorical_accuracy])
print(model.summary())

model.fit(x=[x_feature, x_prelen], y=y, batch_size=1, epochs=2)
