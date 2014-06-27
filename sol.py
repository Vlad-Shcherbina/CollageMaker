import sys
from timeit import default_timer
import contextlib

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
    def compose(self, data):
        with time_it('loading'):
            target, pos = consume_image_description(data, start=0)
            sources = []
            while pos < len(data):
                s, pos = consume_image_description(data, start=pos)
                sources.append(s)
            assert len(sources) == 200

        with time_it('preprocessing'):
            h, w = target.shape
            target = TargetImage(target)

            ias = map(ImageAbstraction, sources)

        with time_it('solving'):
            result = [-1] * 4 * len(sources)
            used_sources = set()

            k = 7
            for i in range(k):
                y1 = h * i // k
                y2 = h * (i + 1) // k
                for j in range(k):
                    x1 = w * j // k
                    x2 = w * (j + 1) // k

                    ia1 = target.get_abstraction(x1, y1, x2, y2)

                    best = 1e10
                    best_index = None
                    for idx, ia2 in enumerate(ias):
                        if idx in used_sources:
                            continue
                        d = ia1.average_error(ia2)
                        if d < best:
                            best = d
                            best_index = idx

                    used_sources.add(best_index)
                    result[best_index * 4 : best_index * 4 + 4] = [y1, x1, y2 - 1, x2 - 1]

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