import sys
from timeit import default_timer
import contextlib
from math import sqrt

import numpy

from img_lib import *


@contextlib.contextmanager
def time_it(task_name='it'):
    start = default_timer()
    yield
    print>>sys.stderr, '{} took {:.3f}'.format(
        task_name, default_timer() - start)


def consume_image_description(data, start):
    h, w = data[start: start + 2]
    arr = numpy.array(data[start + 2: start + 2 + h * w])
    return arr.reshape((h, w)), start + 2 + h * w


class CollageMaker(object):
    def evaluate_placement(self, placements):
        s = 0.0
        for idx, (x1, y1, x2, y2) in placements.items():
            a = self.scalables[idx].downscale(x2 - x1, y2 - y1)
            s += ((self.target.arr[y1:y2, x1:x2] - a)**2).sum()
        s /= self.target.arr.size
        return sqrt(s)

    def grid_placements(self, kw, kh):
        h, w = self.target.arr.shape
        placements = {}
        approx = 0
        for i in range(kh):
            y1 = h * i // kh
            y2 = h * (i + 1) // kh
            for j in range(kw):
                x1 = w * j // kw
                x2 = w * (j + 1) // kw

                ia1 = self.target.get_abstraction(x1, y1, x2, y2)

                best = 1e10
                best_index = None
                for idx, ia2 in enumerate(self.ias):
                    if idx in placements:
                        continue
                    a = self.scalables[idx].arr
                    if a.shape[0] < y2 - y1 or a.shape[1] < x2 - x1:
                        continue
                    d = ia1.average_error(ia2)
                    if d < best:
                        best = d
                        best_index = idx
                if best_index is None:
                    return 1e10, None
                placements[best_index] = (x1, y1, x2, y2)
                approx += best * (x2 - x1) * (y2 - y1)
        return sqrt(1.0 * approx / self.target.arr.size), placements

    def compose(self, data):
        with time_it('loading'):
            target, pos = consume_image_description(data, start=0)
            sources = []
            while pos < len(data):
                s, pos = consume_image_description(data, start=pos)
                sources.append(s)
            assert len(sources) == 200

        with time_it('preprocessing'):
            self.target = target = TargetImage(target)

            self.ias = ias = map(ImageAbstraction, sources)
            self.scalables = map(ScalableImage, sources)

        with time_it():
            ps = [self.grid_placements(kw, kh) for kw in range(1, 8) for kh in range(1, 8)]

        score, placements = min(ps)

        print>>sys.stderr, 'expected score: ', score
        print>>sys.stderr, len(placements)
        result = [-1] * 4 * len(sources)
        for idx, (x1, y1, x2, y2) in placements.items():
            result[idx * 4 : idx * 4 + 4] = [y1, x1, y2 - 1, x2 - 1]
        return result


def main():
    n = int(raw_input())
    data = [int(raw_input()) for _ in range(n)]
    result = CollageMaker().compose(data)
    #print>>sys.stderr, result, len(result)
    for x in result:
        print x
    sys.stdout.flush()


if __name__ == '__main__':
    main()
