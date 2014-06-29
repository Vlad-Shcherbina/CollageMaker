from math import sqrt
import numpy
import numpy.linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import img_lib


a = 1.0 / numpy.array([
    [1, 2, 2],
    [2, 3, 4],
    [2, 4, 3]
])

e, vs = numpy.linalg.eigh(a)
vs *= numpy.sqrt(e)
if vs.sum() < 0:
    vs = -vs
#print vs.T

assert numpy.allclose(numpy.dot(vs, vs.T), a)

def ia_to_sim_space(ia):
    return numpy.dot(vs.T, numpy.array([ia.alpha, ia.alphax, ia.alphay]))


if __name__ == '__main__':

    xs = []
    ys = []
    zs = []
    ss = []
    ns = []

    for i in range(1000):
        #print i
        arr = img_lib.load_image('data/100px/{}.png'.format(i))
        ia = img_lib.ImageAbstraction(arr)
        x, y, z = ia_to_sim_space(ia)

        xs.append(x)
        ys.append(y)
        zs.append(z)
        ss.append(1000.0/sqrt(ia.noise))
        ns.append(sqrt(ia.noise))

    print sorted(ns)[::111]
    #for i in range(200):
    #    if ns[i] < 25:
    #        print i

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=ss)
    plt.show()

#print numpy.dot(vs.T, vs)
#print numpy.linalg.inv(vs)
