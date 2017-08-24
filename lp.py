import numpy as np
import matplotlib.pyplot as plt
import random as random
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
plt.grid(True)

in01 = np.linspace(-10, 10, 2000)

for i in range(100):
    plt.clf()
    w11 = random.uniform(-2, 2)
    w12 = random.uniform(-2, 2)
    w13 = random.uniform(-2, 2)
    w21 = random.uniform(-2, 2)
    w22 = random.uniform(-2, 2)
    w23 = random.uniform(-2, 2)

    b11 = random.uniform(-2, 2)
    b12 = random.uniform(-2, 2)
    b13 = random.uniform(-2, 2)
    b21 = random.uniform(-2, 2)

    out11 = np.tanh(w11 * in01 + b11)
    out12 = np.tanh(w12 * in01 + b12)
    out13 = np.tanh(w13 * in01 + b13)
    out21 = w21 * out11 + w22 * out12 + w23 * out13 + b21

    # plt.plot(in01, out11, c="g")
    # plt.plot(in01, out12, c="y")
    # plt.plot(in01, out13, c="pink")
    plt.plot(in01, out21, c="r")
    # ax.plot(out11, out12, out13, c="b")
    plt.pause(1)
plt.show()
