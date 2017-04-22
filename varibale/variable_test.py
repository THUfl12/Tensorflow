import tensorflow as tf

weights = tf.Variable(tf.random_normal(shape = [1], stddev = 0.35), name = 'weights')
biases = tf.Variable(tf.zeros([200]), name = 'biases')
w2 = tf.Variable(tf.random_normal(shape = [1], stddev = 0.35), name = 'w2')
# before init, assign the values of weights to w2
w2 = w2.assign(weights)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # init all variables
    print(sess.run(weights))
    print(sess.run(w2))
    # after init, assign the values of weights to w2
    w2 = w2.assign(weights)
    print(sess.run(w2))
