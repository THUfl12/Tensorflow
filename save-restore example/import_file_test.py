import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from random import shuffle
import math

def predict_accuracy(y,y_):
    y = y.flatten().astype(int)
    y_ = y_.flatten().astype(int)
    temp = 1-np.mean(np.abs(y-y_))
    return temp

train_data_array = np.loadtxt('./my/train_array.txt', dtype=float)
train_label = np.loadtxt('./my/train_label.txt', dtype=float)

train_label_temp = [[i] for i in train_label]
train_label = train_label_temp
train_label = np.asarray(train_label)


dimension = train_data_array.shape[1]
W = tf.Variable(tf.random_uniform([1,dimension],-1,1))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder('float',[None,dimension])
y_ = tf.placeholder('float',[None,1])

y = tf.sigmoid(tf.matmul(x, tf.transpose(W)) + b)
#
loss = (-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y)))
optimizer = tf.train.FtrlOptimizer(0.001,l1_regularization_strength=0.005,l2_regularization_strength=0.03)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

new_saver = tf.train.Saver()
new_saver.restore(sess, './my/my-model')

for sample_index in range(500,800):
    sess.run(train, {x: train_data_array[sample_index:sample_index + 1, :],
                         y_: train_label[sample_index:sample_index + 1, :]})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 100 == 0:
        print(sample_index, train_W, train_b)

train_y = sess.run(y, {x: train_data_array}) > 0.5
print(len(np.nonzero(W)))
error_rate = predict_accuracy(train_y, train_label)
print(error_rate)

new_saver.save(sess, './my/my-model-2')
for sample_index in range(800,train_data_array.shape[0]):
    sess.run(train, {x: train_data_array[sample_index:sample_index + 1, :],
                         y_: train_label[sample_index:sample_index + 1, :]})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 100 == 0:
        print(sample_index, train_W, train_b)

train_y = sess.run(y, {x: train_data_array}) > 0.5
print(len(np.nonzero(W)))
error_rate = predict_accuracy(train_y, train_label)
print(error_rate)
