import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
x_data = [[1,3],[1,4],[1,5],[1,8],[2,4],[3,3],[3,5],[3,6],[4,4],[6,6],[9,9],[12,12]]
y_data = [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]
x1 = [1,1,1,1,2,3,3,3,3,3,4,4,4,4,6,6,6,9,9,9,12,12]
x2 = [3,4,5,8,4,1,2,3,5,6,1,2,3,4,3,5,6,7,8,9,11,12]
z  = [1,1,1,1,1,0,0,1,1,1,0,0,0,1,0,0,1,0,0,1,0 ,1 ]
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 2])
# add hidden layer
l1 = add_layer(xs, 2, 20, activation_function=tf.nn.relu)
l2 = add_layer(l1, 20, 20, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l2, 20, 1, activation_function=tf.nn.relu)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.003).minimize(loss)
# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# # plot the real data
# fig = plt.figure()
# ax = Axes3D(fig)
# # xm,ym = np.meshgrid(x1,x2)
# ax.plot(x1, x2, z, "ro")
# # plt.ylabel("computer")
# # plt.xlabel("store")
# # plt.grid(True)
# # plt.plot(x1,x2,'ro')
# plt.show()
# x_test = np.linspace(-100, 100, 2000).reshape(1000, 2)
# print(x_test)
prediction_value = sess.run(prediction, feed_dict={xs: x_data})
print(prediction_value)
prediction_value_test = sess.run(prediction, feed_dict={xs: [[2,3],[5,4],[5,5],[5,6],[8,7],[8,8],[8,10],[10,9],[10,10],[10,13],[50,49],[50,50],[50,66]]})
print(prediction_value_test)
# for i in range(30000):
#     # training
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # if i % 1000 == 0:
    #     # to visualize the result and improvement
    #     try:
    #         plt.clf()
    #     except Exception:
    #         pass
    #     prediction_value = sess.run(prediction, feed_dict={xs: [[2,3],[5,4],[5,5],[5,6],[8,7],[8,8],[8,10],[10,9],[10,10],[10,13],[50,49],[50,50],[50,66]]})
    #     expectation =                                           [1,    0,    1,    1,    0,    1,    1,     0,     1,      1,      0,      1]
    #     step = 0
    #     print("epoch:", i + 1000)
    #     print(prediction_value)
    #     plt.grid(True)
    #     plt.xlabel("Test Data Index")
    #     plt.ylabel("Prediction")
    #     plt.plot(prediction_value, 'ro')
    #     plt.pause(0.01)
# plt.show()
