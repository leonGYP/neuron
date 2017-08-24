import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # biases = tf.Variable(tf.random_normal([1, out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
x_data = np.linspace(-10, 10, 200)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.linspace(-10,10,40)[:, np.newaxis]
y_data = 3 * x_data + 5
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 2, activation_function=tf.nn.sigmoid)
# l2 = add_layer(l1, 2, 1, activation_function=tf.nn.sigmoid)
# l3 = add_layer(l2, 100, 100, activation_function=None)
# add output layer
prediction = add_layer(l1, 2, 1, activation_function=None)

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
# important step
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# # plot the real data
plt.ylabel("Y")
plt.xlabel("X")
plt.grid(True)
plt.plot(x_data, y_data, 'ro')
plt.show()


for i in range(30000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 300 == 0:
        # to visualize the result and improvement
        try:
            plt.clf()
        except Exception:
            pass
        x_test = np.linspace(-10, 10, 10000)[:, np.newaxis]
        prediction_value = sess.run(prediction, feed_dict={xs: x_test})
        print(prediction_value)
        print("epoch:", i)
        plt.grid(True)
        plt.plot(x_data, y_data, 'ro', c='r')
        plt.plot(x_test, prediction_value, 'r-', c='b')
        plt.pause(0.01)
print(tf.trainable_variables())
print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
plt.show()
