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


class ScalableImage(object):
    def __init__(self, arr):
        self.arr = arr
        self.rect_sum = RectSum(arr)

    def downscale(self, new_w, new_h):
        arr = self.arr
        h, w = arr.shape
        assert 1 <= new_w <= w
        assert 1 <= new_h <= h

        result = numpy.zeros((new_h, new_w), dtype=int)
        for y in range(new_h):
            for x in range(new_w):
                i1 = y * h // new_h + 1
                i2 = ((y + 1) * h - 1) // new_h
                j1 = x * w // new_w + 1
                j2 = ((x + 1) * w - 1) // new_w
                assert y * h <= i1 * new_h
                assert i2 * new_h <= (y + 1) * h
                assert x * w <= j1 * new_w
                assert j2 * new_w <= (x + 1) * w

                # inner part
                s = self.rect_sum(j1, i1, j2, i2) * new_w * new_h

                # corners
                s += (i1 * new_h - y * h) * (j1 * new_w - x * w) * arr[i1 - 1, j1 - 1]
                s += (i1 * new_h - y * h) * ((x + 1) * w - j2 * new_w) * arr[i1 - 1, j2]
                s += ((y + 1) * h - i2 * new_h) * (j1 * new_w - x * w) * arr[i2, j1 - 1]
                s += ((y + 1) * h - i2 * new_h) * ((x + 1) * w - j2 * new_w) * arr[i2, j2]

                # edges
                s += (i1 * new_h - y * h) * new_w * self.rect_sum(j1, i1 - 1, j2, i1)
                s += ((y + 1) * h - i2 * new_h) * new_w * self.rect_sum(j1, i2, j2, i2 + 1)
                s += new_h * (j1 * new_w - x * w) * self.rect_sum(j1 - 1, i1, j1, i2)
                s += new_h * ((x + 1) * w - j2 * new_w) * self.rect_sum(j2, i1, j2 + 1, i2)

                result[y, x] = (s + h * w // 2) // (w * h)

        return result


class ImageAbstraction(object):
    def __init__(self, arr=None):
        if arr is None:
            self.alphax = 0
            self.alphay = 0
            return

        h, w = arr.shape

        xs = numpy.arange(0.5 / w, 1.0, 1.0 / w).reshape(1, w)
        ys = numpy.arange(0.5 / h, 1.0, 1.0 / h).reshape(h, 1)

        n = arr.size
        sc = arr.sum()
        sx = xs.sum() * h
        sy = ys.sum() * w
        sx2 = (xs**2).sum() * h
        sy2 = (ys**2).sum() * w
        sxy = (xs * ys).sum()

        scx = (arr * xs).sum()
        scy = (arr * ys).sum()

        a = numpy.array([
            [n, sx, sy],
            [sx, sx2, sxy],
            [sy, sxy, sy2]])
        a_inv = numpy.linalg.pinv(a)

        b = numpy.array([sc, scx, scy])
        sol = numpy.dot(a_inv, b)
        self.alpha, self.alphax, self.alphay = sol

        self.noise = ((self.alpha + xs * self.alphax + ys * self.alphay - arr)**2).sum() / arr.size

    def instantiate(self, w, h):
        result = numpy.zeros((h, w))
        for i in range(h):
            for j in range(w):
                q = (self.alpha +
                    (j + 0.5) / h * self.alphax +
                    (i + 0.5) / w * self.alphay)
                c = random.normalvariate(q, sqrt(self.noise))
                c = int(c + 0.5)
                # if c < 0:
                #     c = 0
                # if c > 255:
                #     c = 255
                result[i, j] = c
        return result

    def average_error(self, other):
        a = self.alpha - other.alpha
        ax = self.alphax - other.alphax
        ay = self.alphay - other.alphay
        result = a**2 + (ax**2 + ay**2) * 1.0/3 + a * (ax + ay) + 0.5 * ax * ay
        return result + self.noise + other.noise


def average_error(arr1, arr2):
    assert arr1.shape == arr2.shape
    return 1.0 * ((arr1 - arr2)**2).sum() / arr1.size


class TargetImage(object):
    def __init__(self, arr):
        self.arr = arr
        self.rect_sum = RectSum(arr)
        self.rect_sum2 = RectSum(arr**2)
        h, w = arr.shape
        xs = numpy.array(range(w)).reshape(1, w)
        ys = numpy.array(range(h)).reshape(h, 1)
        self.rect_sum_x = RectSum(arr * xs)
        self.rect_sum_y = RectSum(arr * ys)

    def get_abstraction(self, x1, y1, x2, y2):
        area = (x2 - x1) * (y2 - y1) * 1.0
        s = self.rect_sum(x1, y1, x2, y2)
        s2 = self.rect_sum2(x1, y1, x2, y2)
        result = ImageAbstraction()
        average = result.alpha = s / area

        result.noise = (s2 - 2 * average * s) / area + average**2
        return result


if __name__ == '__main__':
    w = 50
    h = 50

    arr1 = load_image('data/100px/105.png')
    ia1 = TargetImage(arr1).get_abstraction(0, 0, w, h)
    arr1 = arr1[:h, :w]
    #arr1 = ScalableImage(arr1).downscale(w, h)

    arr2 = load_image('data/100px/104.png')
    #h, w = arr2.shape
    arr2 = ScalableImage(arr2).downscale(w, h)

    #ia1 = ImageAbstraction(arr1)
    ia2 = ImageAbstraction(arr2)

    print 'average error', sqrt(average_error(arr1, arr2))
    print 'predicted', sqrt(ia1.average_error(ia2))

    for _ in range(10):
        print 'sample', sqrt(average_error(ia1.instantiate(w, h), ia2.instantiate(w, h)))

    save_image(arr1, 'hz.png')
    save_image(arr2, 'hz2.png')
    # save_image(ia1.instantiate(w, h), 'hz.png')
    # save_image(ia2.instantiate(w, h), 'hz2.png')
