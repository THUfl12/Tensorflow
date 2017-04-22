import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from random import shuffle

def predict_accuracy(y,y_):
    y = y.flatten().astype(int)
    y_ = y_.flatten().astype(int)
    temp = 1-np.mean(np.abs(y-y_))
    return temp

# read csv file from pandas
train_label = pd.read_csv('train_label.txt', sep=' ', header=None)
train_label.columns = ['label']

# get the index of class 1 and 2
index_label_1 = train_label[train_label['label'] == 1].index.tolist()
index_label_2 = train_label[train_label['label'] == 2].index.tolist()

index_label = index_label_1 + index_label_2
shuffle(index_label)
train_label_temp = train_label['label'][index_label]
train_label_temp = train_label_temp - 1

train_label = []
for item in train_label_temp:
    train_label.append([item])

train_label = np.asarray(train_label)

# read train data file from pandas and get the maximum of word index
train_temp_data = pd.read_csv("train.txt", sep=' ', header=None)
train_temp_data.columns = ['docId', 'wordId', 'count']

train_data_index = np.arange(train_temp_data.shape[0])[
    np.in1d(train_temp_data['docId'], index_label)]

train_data = train_temp_data.loc[train_data_index, :]
max_wordId = max(train_data['wordId'])

# convert the dataframe to matrix for convenience
train_data = train_data.as_matrix()

train_data_array = np.zeros((len(index_label),max_wordId), dtype=np.int16)

for row in index_label:
    row_index = train_data[:,0] == row+1
    word_index = train_data[row_index,1]
    train_data_array[row, word_index-1] = train_data[row_index,2]

dimension = train_data_array.shape[1]

# define the placeholder
x = tf.placeholder('float',[None,dimension])
y_ = tf.placeholder('float',[None,1])

# define the variable of the model
W = tf.Variable(tf.random_uniform([1,dimension],-1,1))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(x, tf.transpose(W)) + b)
y = tf.clip_by_value(y,1e-10,1-1e-10)

loss = (-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y)))

optimizer = tf.train.FtrlOptimizer(0.001,l1_regularization_strength=0.005,l2_regularization_strength=0.03)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# if you want to save part of variable, just add to collection
# tf.add_to_collection('W', W)
# tf.add_to_collection('b', b)
# tf.add_to_collection('loss', loss)
# tf.add_to_collection('optimizer', optimizer)
# tf.add_to_collection('train', train)

# before session start, define the saver
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

for sample_index in range(500):
    sess.run(train, {x: train_data_array[sample_index:sample_index + 1, :],
                         y_: train_label[sample_index:sample_index + 1, :]})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 100 == 0:
        print(sample_index, train_W, train_b,
              sess.run(loss / len(index_label), {x: train_data_array, y_: train_label}))

train_y = sess.run(y, {x: train_data_array}) > 0.5
print(len(np.nonzero(train_W)[0]))
error_rate = predict_accuracy(train_y, train_label)
print(error_rate)

np.savetxt('./my/train_array.txt', train_data_array, fmt = '%d')
np.savetxt('./my/train_label.txt', train_label, fmt = '%d')
#
# save all the variables of the session
# path should be '.my-model', don't forget the '.'. In this example, I create a folder named 'my' to store the model and file
saver.save(sess, './my/my-model')


for sample_index in range(500,train_data_array.shape[0]):
    sess.run(train, {x: train_data_array[sample_index:sample_index + 1, :],
                         y_: train_label[sample_index:sample_index + 1, :]})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 100 == 0:
        print(sample_index, train_W, train_b,
              sess.run(loss / len(index_label), {x: train_data_array, y_: train_label}))


train_y = sess.run(y, {x: train_data_array}) > 0.5
print(len(np.nonzero(train_W)[0]))
error_rate = predict_accuracy(train_y, train_label)
print(error_rate)

