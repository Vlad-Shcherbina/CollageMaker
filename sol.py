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

PENALTY = 1e9


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
        self.widths = numpy.array([ia.width for ia in ias])
        self.heights = numpy.array([ia.height for ia in ias])
        self.min_width = self.widths.min()
        self.min_height = self.heights.min()
        print>>sys.stderr, self.min_width, self.min_height
        self.coords = numpy.array([
            list(ia.sim_coords) + [ia.noise]
            for ia in ias])

    def find_nearest(self, ia1, min_w, min_h, used_penalty):
        pos = numpy.zeros(4)
        pos[:3] = ia1.sim_coords
        dists = self.coords - pos
        dists = numpy.einsum('ij,ij->i', dists, dists)
        dists += used_penalty
        if min_w > self.min_width:
            dists[self.widths < min_w] = PENALTY
        if min_h > self.min_height:
            dists[self.heights < min_h] = PENALTY
        idx = dists.argsort()[0]
        return idx, dists[idx] + ia1.noise * ia1.noise


class CollageMaker(object):
    def evaluate_placement(self, placements):
        s = 0.0
        for idx, (x1, y1, x2, y2) in placements.items():
            a = self.scalables[idx].downscale(x2 - x1, y2 - y1)
            s += ((self.target.arr[y1:y2, x1:x2] - a)**2).sum()
        s /= self.target.arr.size
        return sqrt(s)

    def try_subdivide(self, placements, used_penalty, idx):
        x1, y1, x2, y2 = placements[idx]

        ia1 = self.target.get_abstraction(x1, y1, x2, y2)
        penalty = ia1.average_error(self.ias[idx])

        del placements[idx]
        used_penalty[idx] = 0

        best_score = penalty
        best_left = None
        best_right = None
        best_sub = None

        subdivisions = []
        for x in range(x1 + 3, x2 - 3 + 1, 2):
            subdivisions.append(((x1, y1, x, y2), (x, y1, x2, y2)))
        for y in range(y1 + 3, y2 - 3 + 1, 2):
            subdivisions.append(((x1, y1, x2, y), (x1, y, x2, y2)))

        for rect_left, rect_right in subdivisions:
            ia_left = self.target.get_abstraction(*rect_left)
            ia_right = self.target.get_abstraction(*rect_right)

            idx_left, score_left = self.ia_table.find_nearest(
                ia_left,
                min_w=rect_left[2] - rect_left[0],
                min_h=rect_left[3] - rect_left[1],
                used_penalty=used_penalty)
            if idx_left is None:
                continue
            placements[idx_left] = ()
            used_penalty[idx_left] = PENALTY
            idx_right, score_right = self.ia_table.find_nearest(
                ia_right,
                min_w=rect_right[2] - rect_right[0],
                min_h=rect_right[3] - rect_right[1],
                used_penalty=used_penalty)
            del placements[idx_left]
            used_penalty[idx_left] = 0

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
            used_penalty[idx] = PENALTY
            return False
        else:
            assert best_left is not None
            assert best_right is not None
            assert best_sub is not None
            print>>sys.stderr, 'subdividing'
            placements[best_left] = best_sub[0]
            placements[best_right] = best_sub[1]
            used_penalty[best_left] = PENALTY
            used_penalty[best_right] = PENALTY
            return True

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

        h, w = target.arr.shape
        placements = {}
        used_penalty = numpy.zeros((200,))

        with time_it('build partition'):
            items = build_partition(target)
        items.sort(key=lambda (rect, ia): area(*rect), reverse=True)
        with time_it('find_nearest'):
            for (x1, y1, x2, y2), ia in items:
                idx, _ = self.ia_table.find_nearest(
                    ia, x2 - x1, y2 - y1,
                    used_penalty)
                placements[idx] = (x1, y1, x2, y2)
                used_penalty[idx] = PENALTY

        with time_it('subdividing'):
            frozen = set()
            for i in range(10):
                print>>sys.stderr, 'level', i
                # TODO: order in decreasing score improvement, not area.
                for idx, _ in sorted(placements.items(), key=lambda (k, v): -area(*v)):
                    if idx in frozen:
                        continue
                    if default_timer() - GLOBAL_START > TIME_LIMIT:
                        break
                    if not self.try_subdivide(placements, used_penalty, idx):
                        frozen.add(idx)

        for i, p in enumerate(used_penalty):
            if p == 0:
                assert i not in placements
            elif p == PENALTY:
                assert i in placements
            else:
                assert False, p
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
