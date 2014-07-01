import sys

import numpy

from img_lib import *


class Mipmap(object):
    def __init__(self, arr):
        self.arr = arr
        self.si = ScalableImage(arr)

        h, w = arr.shape

        def intermediate_sizes(size):
            result = []
            x = 1
            while x < size * 2 // 3:
                result.append(x)
                x *= 2
            result.append(size)
            return result

        self.ws = intermediate_sizes(w)
        self.hs = intermediate_sizes(h)

        self.ia = {}
        for w in self.ws:
            for h in self.hs:
                if (h, w) == arr.shape:
                    d = arr
                else:
                    d = self.si.downscale(w, h)
                self.ia[h, w] = abstraction_from_arr(d)
                #print>>sys.stderr, '{:4.1f}'.format(self.ia[h, w].noise),
            #print>>sys.stderr

    def render(self):
        result = numpy.zeros((sum(self.hs), sum(self.ws)))
        x = 0
        for w in self.ws:
            y = 0
            for h in self.hs:
                result[y:y+h, x:x+w] = self.si.downscale(w, h)
                y += h
            x += w

        return result


if __name__ == '__main__':
    arr = load_image('data/100px/1.png')

    h, w = arr.shape
    print w, h

    m = Mipmap(arr)

    save_image(m.render(), 'mip.png')
