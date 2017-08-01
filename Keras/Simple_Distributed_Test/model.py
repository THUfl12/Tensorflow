from keras.layers import Dense, Input, concatenate, Reshape, Embedding, Dropout, GRU
from keras.models import Model
from keras.layers.core import Lambda
from tensorflow.python.ops.control_flow_ops import with_dependencies
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K

# use cart weight to average LSTM output
def cart_weight(input):
    rnn_state = input[0]
    state_weight = input[1]
    sum_weight = K.sum(state_weight, 1)
    sum_weight = K.expand_dims(sum_weight, 1)
    weight = K.expand_dims(state_weight / sum_weight, 2)
    LSTM_average = K.squeeze(K.batch_dot(rnn_state, weight, 1), 2)
    return LSTM_average

# use cart weight to expand the embedded vector presentation of feature
# for example, if embedding size is (None, 10, 3), after adding the cart weight,
# shape will be (None, 10, 4)
def expand_feature(input):
    emb_state = input[0]
    cart_flag = input[1]
    exp_weight = K.expand_dims(cart_flag, 2)
    return K.concatenate([emb_state, exp_weight], 2)

# define flags
tf.app.flags.DEFINE_string("ps_hosts", "10.1.8.27:2223",
                           "comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "10.1.8.27:2222",
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    feature_len = 3
    feature_dim = 2
    feature_pre_len = 2
    feature_purchase_len = 5
    feature_cart_len = 5
    units = 8
    embedding_size = 3
    feature_size = 4

    x_cart = np.array([[1, 2, 3, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 3, 4, 5, 6, 7, 8, 9]])
    x_purchase = np.array([[1, 2, 3, 3, 4], [2, 3, 1, 4, 1]])
    x_feature = np.array([[1, 1, 2, 2, 1, 2], [2, 3, 3, 3, 4, 3]])
    x_N = np.array([[2, 3], [4, 3]])
    y = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            K.set_learning_phase(1)
            K.manual_variable_initialization(True)
            
            # bulid model use Keras
            # LSTM pre process for one feature
            embedding_layer = Embedding(output_dim=embedding_size,
                                        input_dim=feature_size + 1,
                                        name='feature_embedding')

            feature_p = Input(shape=(feature_purchase_len,), name='feature_purchase_input')
            feature_p_embedding = embedding_layer(feature_p)

            feature_input = Input(shape=(feature_len * feature_dim,), name='feature_input')
            feature_input_embedding = embedding_layer(feature_input)
            feature_input_embedding = Reshape((feature_len, feature_dim, embedding_size), name='reshape_feature_input')(
                feature_input_embedding)
            feature_average = Lambda(lambda x: K.mean(x, axis=2), name='mean_feature_input_embedding')(
                feature_input_embedding)

            feature_N = Input(shape=(feature_pre_len,), name='feature_N_input')
            feature_N_embedding = embedding_layer(feature_N)

            feature_p_s_embedding = concatenate([feature_p_embedding, feature_average, feature_N_embedding], axis=1)

            feature_weight = Input(shape=(feature_purchase_len + feature_len + feature_pre_len,), name='weight_input')

            feature_p_s_embedding = Lambda(expand_feature, name='expand_weight')([feature_p_s_embedding, feature_weight])

            # feature RNN unit (LSTM or GRU), if expand feature dimensions, return_sequence is False
            LSTM_layer = GRU(units=units, return_sequences=False, name='feature_LSTM')

            # if average output state of GRU, return_sequences is True
            # LSTM_layer = GRU(units=units, return_sequences=True, name='feature_LSTM')
            LSTM_dropout = Dropout(0.1, name='LSTM_dropout')

            LSTM_p_s = LSTM_layer(feature_p_s_embedding)
            LSTM_state = LSTM_dropout(LSTM_p_s)

            # use cart weight to average LSTM outputs
            # LSTM_p_s = LSTM_dropout(LSTM_p_s)
            # LSTM_state = Lambda(cart_weight, name='cart_weight')([LSTM_p_s, feature_weight])

            output = Dense(feature_size, activation='softmax', name='feature_output')(LSTM_state)

            model = Model(inputs=[feature_p, feature_input, feature_N, feature_weight],
                          outputs=[output])
            # model built
            # define train label and loss function, since it is a dirstributed system, we use global_step to record the process
            targets = tf.placeholder(shape=(None, feature_size), dtype=tf.float32)
            loss = tf.reduce_mean(keras.losses.categorical_crossentropy(targets, output))
            global_step = tf.Variable(0, name='global_step', trainable=False)
            
            # define optimizer
            optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0, epsilon=1e-6)

            with tf.control_dependencies(model.updates):
                barrier = tf.no_op(name='update_barrier')

            # Compute gradients of `loss` for the variables in `var_list`. This is the first part of `minimize()`. 
            with tf.control_dependencies([barrier]):
                grads = optimizer.compute_gradients(loss,
                                                    model.trainable_weights,
                                                    gate_gradients=tf.train.Optimizer.GATE_OP,
                                                    aggregation_method=None,
                                                    colocate_gradients_with_ops=False)
            # Apply gradients to variables. This is the second part of `minimize()`. It returns an `Operation` that
            # applies gradients.
            grad_update = optimizer.apply_gradients(grads, global_step=global_step)
            train_tensor = with_dependencies([grad_update],
                                             loss,
                                             name='train')

            print(model.summary())

            saver = tf.train.Saver()
            summary = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir="./checkpoint",
                                     init_op=init_op,
                                     summary_op=summary,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600)

            with sv.managed_session(server.target) as sess:
                step = 0
                while not sv.should_stop() and step < 100:
                    train_feed = {feature_p: x_purchase, feature_input: x_feature, feature_N: x_N,
                                  feature_weight: x_cart, targets: y}
                    _, loss_value, step_value = sess.run([train_tensor, loss, global_step],
                                                         feed_dict=train_feed)
                    step += 1
                    if step % 10 == 0:
                        print([loss_value, step_value+1])

                test_feed = {feature_p: x_purchase, feature_input: x_feature, feature_N: x_N, feature_weight: x_cart}
                test_result = sess.run(output, test_feed)
                print(test_result)

            sv.stop()

if __name__ == '__main__':

    tf.app.run()
