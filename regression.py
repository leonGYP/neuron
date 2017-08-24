import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = [[1], [2], [3], [4], [5]]
y_data = [[1], [2], [3], [4], [5]]
x_test = np.linspace(0, 6, 10)[:, np.newaxis]
y_test = x_test * - 0.2 + 2
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
prediction = add_layer(xs, 1, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# # plot the real data
plt.ylabel("X")
plt.xlabel("Y")
plt.grid(True)
plt.xlim(0.0, 6.0)
plt.ylim(0.0, 6.0)
plt.plot(x_data, y_data, 'ro', c='r')
plt.plot(x_test, y_test, 'r-', c='g')
plt.show()


for i in range(100):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 1 == 0:
        # to visualize the result and improvement
        try:
            plt.clf()
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_test})
        print(prediction_value)
        plt.grid(True)
        plt.ylabel("X2")
        plt.xlim(0.0, 6.0)
        plt.ylim(0.0, 6.0)
        plt.xlabel("X1")
        plt.plot(x_data, y_data, 'ro', c='r')
        plt.plot(x_test, prediction_value, 'r-', c='g')
        plt.pause(0.2)
plt.show()
