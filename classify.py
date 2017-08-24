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


x_data = [[1,1],[1,2],[1,3],[2,1],[2,2],[3,1],[1,5],[2,4],[2,5],[3,3],[3,4],[3,5]]
x11 = [1, 1, 1, 2, 2, 3]
x12 = [1, 2, 3, 1, 2, 1]
x21 = [1, 2, 2, 3, 3, 3]
x22 = [5, 4, 5, 3, 4, 5]
y_data = [[-1],[-1],[-1],[-1],[-1],[-1],[1],[1],[1],[1],[1],[1]]
y1 = [-1, -1, -1, -1, -1, -1]
y2 = [ 1,  1,  1,  1,  1,  1]
xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])
prediction = add_layer(xs, 2, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# # plot the real data
plt.ylabel("X2")
plt.xlabel("X1")
plt.grid(True)
plt.xlim(0.0, 4.0)
plt.ylim(0.0, 6.0)
plt.plot(x11, x12, 'ro', c='r')
plt.plot(x21, x22, 'ro', c='b')
plt.show()


for i in range(10000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 200 == 0:
        # to visualize the result and improvement
        try:
            plt.clf()
        except Exception:
            pass
        x_test = np.linspace(0, 4, 10)
        vri = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        for i in range(len(prediction_value)):
            if prediction_value[i] >= 0:
                prediction_value[i] = 1
            else:
                prediction_value[i] = -1
        print(prediction_value)
        print('w1:',vri[0][0])
        print('w2:',vri[0][1])
        print('b:',vri[1])
        y = (vri[0][0] * x_test + vri[1][0]) / (- vri[0][1])
        plt.grid(True)
        plt.ylabel("X2")
        plt.xlim(0.0, 4.0)
        plt.ylim(0.0, 6.0)
        plt.xlabel("X1")
        plt.plot(x11, x12, 'ro', c='r')
        plt.plot(x21, x22, 'ro', c='b')
        plt.plot(x_test, y, 'r-', c='g')
        plt.pause(0.2)
plt.show()
