from timeit import default_timer
GLOBAL_START = default_timer()

import sys
from math import sqrt
import random

try:
    import Image
except ImportError:
    # to work in TC environment
    print>>sys.stderr, "warning: can't import Image module"
import numpy
import numpy.linalg

from stats import *


def load_image(path):
    img = Image.open(path)
    return numpy.array(img.getdata()).reshape(img.size[1], img.size[0])


def save_image(arr, path):
    Image.fromarray(arr.astype(numpy.uint8)).save(path)


def naive_downscale(arr, new_w, new_h):
    result = numpy.zeros((new_h, new_w), dtype=int)
    h, w = arr.shape
    assert new_w <= w
    assert new_h <= h
    for y in range(new_h):
        for x in range(new_w):
            s = 0
            for yy in range(h):
                for xx in range(w):
                    s += arr[(y * h + yy) // new_h, (x * w + xx) // new_w]
            result[y, x] = (s + h * w // 2) // (h * w)
    return result


class RectSum(object):
    def __init__(self, arr):
        h, w = arr.shape
        self.cum = numpy.zeros((h + 1, w + 1), dtype=int)
        self.cum[1:, 1:] = numpy.cumsum(numpy.cumsum(arr, axis=0), axis=1)
    def __call__(self, x1, y1, x2, y2):
        cum = self.cum
        return cum[y2, x2] - cum[y1, x2] - cum[y2, x1] + cum[y1, x1]


class CornerRectSum(object):
    def __init__(self, arr):
        self.h, self.w = arr.shape
        self.sum = arr.sum()

    def __call__(self, x1, y1, x2, y2):
        assert x1 == y1 == 0
        assert x2 == self.w
        assert y2 == self.h
        return self.sum


class ScalableImage(object):
    def __init__(self, arr):
        self.arr = arr
        self.rect_sum = RectSum(arr)

    def downscale(self, new_w, new_h):
        arr = self.arr
        h, w = arr.shape

        xs = numpy.arange(w, (new_w + 1) * w, w)
        x_fracs = xs % new_w
        xs //= new_w
        xs[-1] -= 1
        x_fracs[-1] += new_w

        ys = numpy.arange(h, (new_h + 1) * h, h)
        y_fracs = ys % new_h
        ys //= new_h
        ys[-1] -= 1
        y_fracs[-1] += new_h

        ys = ys.reshape(new_h, 1)
        y_fracs = y_fracs.reshape(new_h, 1)

        q = new_w * new_h * self.rect_sum.cum[ys, xs]
        q += (self.rect_sum.cum[ys, xs + 1] - self.rect_sum.cum[ys, xs]) * x_fracs * new_h
        q += (self.rect_sum.cum[ys + 1, xs] - self.rect_sum.cum[ys, xs]) * y_fracs * new_w
        q += arr[ys, xs] * x_fracs * y_fracs

        q[:, 1:] -= q[:, :-1].copy()
        q[1:, :] -= q[:-1, :].copy()
        q = (q + (h * w) // 2) // (w * h)

        return q


# Precompute transformation to similarity space
a = 1.0 / numpy.array([
    [1, 2, 2],
    [2, 3, 4],
    [2, 4, 3]
])
e, vs = numpy.linalg.eigh(a)
vs *= numpy.sqrt(e)
if vs.sum() < 0:
    vs = -vs
to_sim_matrix = vs.T
assert numpy.allclose(numpy.dot(to_sim_matrix.T, to_sim_matrix), a)


class ImageAbstraction(object):
    def compute_alphas(self):
        self.alpha, self.alphax, self.alphay = \
            numpy.linalg.inv(to_sim_matrix).dot(self.sim_coords)

    def instantiate(self, w, h):
        self.compute_alphas()
        result = numpy.zeros((h, w))
        for i in range(h):
            for j in range(w):
                q = (self.alpha +
                    (j + 0.5) / w * self.alphax +
                    (i + 0.5) / h * self.alphay)
                c = random.normalvariate(q, self.noise)
                c = int(c + 0.5)
                if c < 0:
                    c = 0
                if c > 255:
                    c = 255
                result[i, j] = c
        return result

    def naive_average_error(self, other):
        self.compute_alphas()
        other.compute_alphas()
        a = self.alpha - other.alpha
        ax = self.alphax - other.alphax
        ay = self.alphay - other.alphay
        result = a**2 + (ax**2 + ay**2) * 1.0/3 + a * (ax + ay) + 0.5 * ax * ay
        return result + self.noise**2 + other.noise**2

    def average_error(self, other):
        d = self.sim_coords - other.sim_coords
        result = d.dot(d) + self.noise**2 + other.noise**2
        #assert abs(self.naive_average_error(other) - result) < 1e-6
        return result

    def __str__(self):
        self.compute_alphas()
        return 'IA({:.1f}, {:.1f}, {:.1f})+-{:.1f}'.format(
            self.alpha, self.alphax, self.alphay, self.noise)


def average_error(arr1, arr2):
    assert arr1.shape == arr2.shape
    return 1.0 * ((arr1 - arr2)**2).sum() / arr1.size


class Namespace(object):
    pass


class TargetImage(object):
    def __init__(self, arr, rect_sum_factory=RectSum):
        self.arr = arr
        self.rect_sum = rect_sum_factory(arr)
        self.rect_sum2 = rect_sum_factory(arr**2)
        h, w = arr.shape
        xs = numpy.arange(w).reshape(1, w)
        ys = numpy.arange(h).reshape(h, 1)
        self.rect_sum_x = rect_sum_factory(arr * xs)
        self.rect_sum_y = rect_sum_factory(arr * ys)
        self._cache = {}

    _global_cache = {}



    @TimeIt('get_abstraction')
    def get_abstraction(self, x1, y1, x2, y2):
        rect = x1, y1, x2, y2
        if rect in self._cache:
            add_value('get_abstraction from cache', 1)
            return self._cache[rect]

        n = 1.0 * (x2 - x1) * (y2 - y1)
        sc = self.rect_sum(x1, y1, x2, y2)
        sc2 = self.rect_sum2(x1, y1, x2, y2)

        scx = 1.0 * self.rect_sum_x(x1, y1, x2, y2) / (x2 - x1) - sc * (x1 - 0.5) / (x2 - x1)
        scy = 1.0 * self.rect_sum_y(x1, y1, x2, y2) / (y2 - y1) - sc * (y1 - 0.5) / (y2 - y1)

        cache_key = y2 - y1, x2 - x1
        q = self._global_cache.get(cache_key)
        if q is None:
            with TimeIt('get_abstraction matrix'):
                self._global_cache[cache_key] = q = Namespace()

                q.sx = q.sy = (y2 - y1) * (x2 - x1) * 0.5

                q.sx2 = 1.0 * (x2 * (x2 - 1) * (2*x2 - 1) - x1 * (x1 - 1) * (2*x1 - 1)) / (x2 - x1)**2 / 6
                q.sx2 -= 0.5 * (2*x1 - 1) / (x2 - x1)**2 * (x2 * (x2 - 1) - x1 * (x1 - 1))
                q.sx2 += (x1 - 0.5) * (x1 - 0.5) / (x2 - x1)
                q.sx2 *= y2 - y1

                q.sy2 = 1.0 * (y2 * (y2 - 1) * (2*y2 - 1) - y1 * (y1 - 1) * (2*y1 - 1)) / (y2 - y1)**2 / 6
                q.sy2 -= 0.5 * (2*y1 - 1) / (y2 - y1)**2 * (y2 * (y2 - 1) - y1 * (y1 - 1))
                q.sy2 += (y1 - 0.5) * (y1 - 0.5) / (y2 - y1)
                q.sy2 *= x2 - x1

                q.sxy = 0.25 * (x2 * (x2 - 1) - x1 * (x1 - 1)) * (y2 * (y2 - 1) - y1 * (y1 - 1)) / n
                q.sxy -= q.sx * (y1 - 0.5) / (y2 - y1)
                q.sxy -= q.sy * (x1 - 0.5) / (x2 - x1)
                q.sxy -= (x1 - 0.5) * (y1 - 0.5)

                a = numpy.array([
                    [n, q.sx, q.sy],
                    [q.sx, q.sx2, q.sxy],
                    [q.sy, q.sxy, q.sy2]])
                q.a_inv = numpy.linalg.pinv(a)

        ia = ImageAbstraction()

        b = numpy.array([sc, scx, scy])
        sol = q.a_inv.dot(b)
        ia.noise = (sc2 - sol.dot(b)) * (1.0 / n)

        if ia.noise < -1e-3:
            print>>sys.stderr, 'warning: noise < 0', ia.noise
        ia.noise = sqrt(max(0, ia.noise))

        ia.sim_coords = to_sim_matrix.dot(sol)

        ia.width = x2 - x1
        ia.height = y2 - y1

        self._cache[rect] = ia
        return ia


def abstraction_from_arr(arr):
    h, w = arr.shape
    return TargetImage(arr, CornerRectSum).get_abstraction(0, 0, w, h)


if __name__ == '__main__':
    x0 = 3
    y0 = 1
    w = 50
    h = 40

    arr1 = load_image('data/100px/1.png')
    ia1 = TargetImage(arr1).get_abstraction(x0, y0, w, h)
    arr1 = arr1[y0:h, x0:w]

    arr2 = load_image('data/100px/104.png')
    arr2 = ScalableImage(arr2).downscale(w - x0, h - y0)
    ia2 = TargetImage(arr2).get_abstraction(0, 0, w-x0, h-y0)

    print ia1
    print ia2

    print 'average error', sqrt(average_error(arr1, arr2))
    print 'predicted', sqrt(ia1.average_error(ia2))

    for _ in range(10):
        print 'sample', sqrt(average_error(ia1.instantiate(w-x0, h-y0), ia2.instantiate(w-x0, h-y0)))

    save_image(arr1, 'hz.png')
    save_image(arr2, 'hz2.png')
    save_image(ia1.instantiate(w, h), 'hz_i.png')
    save_image(ia2.instantiate(w, h), 'hz2_i.png')
