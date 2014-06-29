from math import sqrt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import img_lib


if __name__ == '__main__':

    xs = []
    ys = []
    zs = []
    ss = []
    ns = []

    for i in range(1000):
        #print i
        arr = img_lib.load_image('data/100px/{}.png'.format(i))
        h, w = arr.shape
        ia = img_lib.TargetImage(arr).get_abstraction(0, 0, w, h)
        x, y, z = ia.sim_coords

        xs.append(x)
        ys.append(y)
        zs.append(z)
        ss.append(1000.0/sqrt(ia.noise))
        ns.append(sqrt(ia.noise))

    print sorted(ns)[::111]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=ss)
    plt.show()
