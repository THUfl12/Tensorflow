from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

embedding_size = 10
feature_size = 50
feature_purchase_len = 100

embedding_layer = Embedding(output_dim=embedding_size,
                            input_dim=feature_size + 1,
                            name='feature_embedding')

feature_p = Input(shape=(feature_purchase_len,), name='feature_input')
# output shape is (None, 100, 10)
feature_p_embedding = embedding_layer(feature_p)

# output shape will be (None, 98, 1)
# [which is (None, newsteps, nb_filters)], since padding style is valid, so the newsteps=100-3+1
# the con_1 parameters is 1*(3*10+1) (first 1 is nb_filters, 3 is kernel_size, 10 is the embedding size, 1 is bias)
con_1 = Conv1D(filters=1, kernel_size=3, padding='valid', activation='relu')(feature_p_embedding)
# output shape will be (None, 49, 1)
con_1 = MaxPooling1D(pool_size=2)(con_1)

model = Model(inputs=[feature_p], outputs=[con_1])
model.compile(optimizer='adam', loss='mse')

print(model.summary())
