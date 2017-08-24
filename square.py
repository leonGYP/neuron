import numpy as np
import matplotlib.pyplot as plt

x_data = np.matrix('1.0;2.0;3.0;4.0;5.0')
y_data = np.matrix('1.0;2.0;3.0;4.0;5.0')
weight = np.matrix('7.0')
bias = np.matrix('1.0')


def bp(x, t, w, b):
    global weight, bias
    p = x * w + b
    weight = weight + 0.3 * np.multiply((t - p), x_data).sum(axis=0) / (2 * len(x_data))
    bias = bias + 0.3 * (t - p).sum(axis=0) / (2 * len(x_data))
    print("weight" + str(weight))
    print("bias" + str(bias))


for i in range(100):
    bp(x_data, y_data, weight, bias)
    plt.clf()
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.grid(True)
    yp = x_data * weight + bias
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, yp, 'r-', c='b')
    plt.pause(0.001)
    print("epoch:" + str(i + 1))
plt.show()
