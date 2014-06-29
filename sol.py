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

    def best_match(self, ia1, min_w, min_h, occupied):
        best_score = 1e10
        best_index = None
        for idx, ia2 in enumerate(self.ias):
            if idx in occupied:
                continue
            a = self.scalables[idx].arr
            if a.shape[0] < min_h or a.shape[1] < min_w:
                continue
            d = ia1.average_error(ia2)
            if d < best_score:
                best_score = d
                best_index = idx
        return best_index, best_score

    def try_subdivide(self, placements, idx):
        x1, y1, x2, y2 = placements[idx]

        ia1 = self.target.get_abstraction(x1, y1, x2, y2)
        penalty = ia1.average_error(self.ias[idx])

        del placements[idx]

        best_score = penalty
        best_left = None
        best_right = None
        best_x = None

        for x in range(x1 + 8, x2 - 8 + 1, 2):
            ia_left = self.target.get_abstraction(x1, y1, x, y2)
            ia_right = self.target.get_abstraction(x, y1, x2, y2)

            idx_left, score_left = self.best_match(ia_left, x - x1, y2 - y1, placements)
            if idx_left is None:
                continue
            placements[idx_left] = ()
            idx_right, score_right = self.best_match(ia_right, x2 - x, y2 - y1, placements)
            del placements[idx_left]

            if idx_right is None:
                continue

            score = 1.0 * (score_left * (x - x1) + score_right * (x2 - x)) / (x2 - x1)
            if score < best_score:
                best_score = score
                best_left = idx_left
                best_right = idx_right
                best_x = x

        if best_left is None:
            placements[idx] = (x1, y1, x2, y2)
        else:
            assert best_left is not None
            assert best_right is not None
            assert best_x is not None
            print>>sys.stderr, 'subdividing'
            placements[best_left] = (x1, y1, best_x, y2)
            placements[best_right] = (best_x, y1, x2, y2)

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
                best_index, best_score = self.best_match(
                    ia1,
                    min_w=x2 - x1, min_h=y2 - y1,
                    occupied=placements)
                if best_index is None:
                    return 1e10, None

                placements[best_index] = (x1, y1, x2, y2)
                approx += best_score * (x2 - x1) * (y2 - y1)
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

            self.ias = ias = []
            for source in sources:
                t = TargetImage(source)
                h, w = source.shape
                self.ias.append(t.get_abstraction(0, 0, w, h))
            self.scalables = map(ScalableImage, sources)

        with time_it('grid placements'):
            ps = [self.grid_placements(kw, kh) for kw in range(1, 8) for kh in range(1, 8)]

        score, placements = min(ps)

        with time_it('subdividing'):
            for idx in set(placements):
                self.try_subdivide(placements, idx)

        print>>sys.stderr, 'expected score: ', score
        print>>sys.stderr, len(placements)
        result = [-1] * 4 * len(sources)
        for idx, (x1, y1, x2, y2) in placements.items():
            result[idx * 4 : idx * 4 + 4] = [y1, x1, y2 - 1, x2 - 1]

        print>>sys.stderr, 'TOTAL TIME: {:.2f}s'.format(default_timer() - GLOBAL_START)
        return result


def main():
    n = int(raw_input())
    data = [int(raw_input()) for _ in range(n)]
    result = CollageMaker().compose(data)
    for x in result:
        print x
    sys.stdout.flush()


if __name__ == '__main__':
    main()
