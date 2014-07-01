import sys
from timeit import default_timer
import contextlib
from math import sqrt
import random

import numpy

from img_lib import *
from target_partition import *
from mipmaps import *


TIME_LIMIT = 9.0


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


class IALookupTable(object):
    def __init__(self, ias):
        self.ias = sorted(enumerate(ias), key=lambda (i, ia): ia.noise)

    def find_nearest(self, ia1, min_w, min_h, used):
        best_score = 1e10
        best_index = None
        for idx, ia2 in self.ias:
            if best_score < ia1.noise**2 + ia2.noise**2:
                break
            if idx in used:
                continue
            if ia2.height < min_h or ia2.width < min_w:
                continue
            d = ia1.average_error(ia2)
            if d < best_score:
                best_score = d
                best_index = idx
        return best_index, best_score


class CollageMaker(object):
    def evaluate_placement(self, placements):
        s = 0.0
        for idx, (x1, y1, x2, y2) in placements.items():
            a = self.scalables[idx].downscale(x2 - x1, y2 - y1)
            s += ((self.target.arr[y1:y2, x1:x2] - a)**2).sum()
        s /= self.target.arr.size
        return sqrt(s)

    def try_subdivide(self, placements, idx):
        x1, y1, x2, y2 = placements[idx]

        ia1 = self.target.get_abstraction(x1, y1, x2, y2)
        penalty = ia1.average_error(self.ias[idx])

        del placements[idx]

        best_score = penalty
        best_left = None
        best_right = None
        best_sub = None

        subdivisions = []
        for x in range(x1 + 5, x2 - 5 + 1, 2):
            subdivisions.append(((x1, y1, x, y2), (x, y1, x2, y2)))
        for y in range(y1 + 5, y2 - 5 + 1, 2):
            subdivisions.append(((x1, y1, x2, y), (x1, y, x2, y2)))

        for rect_left, rect_right in subdivisions:
            ia_left = self.target.get_abstraction(*rect_left)
            ia_right = self.target.get_abstraction(*rect_right)

            idx_left, score_left = self.ia_table.find_nearest(
                ia_left,
                min_w=rect_left[2] - rect_left[0],
                min_h=rect_left[3] - rect_left[1],
                used=placements)
            if idx_left is None:
                continue
            placements[idx_left] = ()
            idx_right, score_right = self.ia_table.find_nearest(
                ia_right,
                min_w=rect_right[2] - rect_right[0],
                min_h=rect_right[3] - rect_right[1],
                used=placements)
            del placements[idx_left]

            if idx_right is None:
                continue

            area_left = (rect_left[2] - rect_left[0]) * (rect_left[3] - rect_left[1])
            area_right = (rect_right[2] - rect_right[0]) * (rect_right[3] - rect_right[1])
            score = 1.0 * (score_left * area_left + score_right * area_right) / (area_left + area_right)
            if score < best_score:
                best_score = score
                best_left = idx_left
                best_right = idx_right
                best_sub = rect_left, rect_right

        if best_sub is None:
            placements[idx] = (x1, y1, x2, y2)
            return False
        else:
            assert best_left is not None
            assert best_right is not None
            assert best_sub is not None
            print>>sys.stderr, 'subdividing'
            placements[best_left] = best_sub[0]
            placements[best_right] = best_sub[1]
            return True

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
                best_index, best_score = self.ia_table.find_nearest(
                    ia1,
                    min_w=x2 - x1, min_h=y2 - y1,
                    used=placements)
                if best_index is None:
                    return 1e10, None

                placements[best_index] = (x1, y1, x2, y2)
                approx += best_score * (x2 - x1) * (y2 - y1)
        return sqrt(1.0 * approx / self.target.arr.size), placements

    def compose(self, data):
        random.seed(42)
        with time_it('loading'):
            target, pos = consume_image_description(data, start=0)
            sources = []
            while pos < len(data):
                s, pos = consume_image_description(data, start=pos)
                sources.append(s)
            assert len(sources) == 200

        with time_it('mipmaps'):
            for source in sources:
                Mipmap(source)

        with time_it('preprocessing'):
            self.target = target = TargetImage(target)

            self.ias = ias = []
            for source in sources:
                t = TargetImage(source)
                h, w = source.shape
                self.ias.append(t.get_abstraction(0, 0, w, h))
            self.ia_table = IALookupTable(ias)
            self.scalables = map(ScalableImage, sources)

        #with time_it('grid placements'):
        #    ps = [self.grid_placements(kw, kh) for kw in range(1, 8) for kh in range(1, 8)]
        #score, placements = min(ps)

        h, w = target.arr.shape
        placements = {}

        with time_it('build partition'):
            items = build_partition(target)
        items.sort(key=lambda (rect, ia): area(*rect), reverse=True)
        with time_it('find_nearest'):
            for (x1, y1, x2, y2), ia in items:
                idx, _ = self.ia_table.find_nearest(ia, x2 - x1, y2 - y1, placements)
                placements[idx] = (x1, y1, x2, y2)

        with time_it('subdividing'):
            frozen = set()
            for i in range(10):
                print>>sys.stderr, 'level', i
                for idx, _ in sorted(placements.items(), key=lambda (k, v): -area(*v)):
                    if idx in frozen:
                        continue
                    if default_timer() - GLOBAL_START > TIME_LIMIT:
                        break
                    if not self.try_subdivide(placements, idx):
                        frozen.add(idx)

        print>>sys.stderr, len(placements)
        result = [-1] * 4 * len(sources)
        for idx, (x1, y1, x2, y2) in placements.items():
            result[idx * 4 : idx * 4 + 4] = [y1, x1, y2 - 1, x2 - 1]

        print>>sys.stderr, 'TOTAL TIME: {:.2f}s'.format(default_timer() - GLOBAL_START)
        sys.stderr.flush()
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
