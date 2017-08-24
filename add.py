# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    biases = tf.Variable(tf.random_normal([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
# l1 = add_layer(xs, 2, 2, activation_function=None)
# l2 = add_layer(l1, 2, 2, activation_function=tf.nn.sigmoid)
# add output layer
prediction = add_layer(xs, 2, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


x_data = []
y_data = []
for i in range(3000):
    # training
    if i < 3:
        x1 = input("第一个数是什么:")
        x2 = input("第二个数是什么:")
        prediction_value = sess.run(prediction, feed_dict={xs: [[x1, x2]]})
        print('我猜结果是')
        print(round(prediction_value))
        rv = input("正确结果是多少?")
        print("\n")
        x_data.append([x1, x2])
        y_data.append([rv])
    if i == 3:
        print('不好玩,让我思考一下!')
    if i % 300 == 0:
        sys.stdout.write("😝 ")
        sys.stdout.flush()
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i == 2995:
        print('\n\n我想好了,再来!\n')
    if i > 2995:
        x1 = input("第一个数是什么:")
        x2 = input("第二个数是什么:")
        prediction_value = sess.run(prediction, feed_dict={xs: [[x1, x2]]})
        print('我猜结果是')
        print(round(prediction_value))
        print("\n")
