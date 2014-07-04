import logging
logger = logging.getLogger('stats')

from math import sqrt
from timeit import default_timer
from functools import wraps

from collections import defaultdict, Counter


class CounterStat(object):
    def __init__(self):
        self.n = 0
        self.counts = Counter()

    def add_value(self, x):
        self.n += 1
        self.counts[x] += 1

    def __str__(self):
        return 'n={}, {}'.format(self.n, self.counts)


def float_to_str(x, precision=1):
    '''
    Convert float to string with at least `precision` significant digits.

    >>> float_to_str(0.123, 2)
    '0.12'
    >>> float_to_str(1.23, 2)
    '1.2'
    >>> float_to_str(0.00123, 2)
    '0.0012'

    We always keep at least one digit after decimal point to indicate that
    it's a float:
    >>> float_to_str(123.45, 2)
    '123.5'

    Trailing zeroes, except the one immediately following decimal point, are removed:
    >>> float_to_str(0, 5)
    '0.0'
    >>> float_to_str(0.12, 5)
    '0.12'
    '''
    q = abs(x)
    if q > 1:
        qq = 1
        while q >= qq and precision > 1:
            qq *= 10
            precision -=1
    else:
        MAX_PRECISION = 10
        while q < 0.1 and precision < MAX_PRECISION:
            q *= 10
            precision += 1
    result = '{0:.{1}f}'.format(x, precision)
    while result.endswith('0') and not result.endswith('.0'):
        assert '.' in result
        result = result[:-1]
    return result


class Stat(object):
    def __init__(self, show_total=False):
        self.show_total = show_total
        self.n = 0
        self.sum = 0
        self.sum2 = 0
        self.max = float('-inf')
        self.min = float('+inf')

    def add_value(self, x):
        self.n += 1
        self.sum += x
        self.sum2 += x*x
        self.max = max(self.max, x)
        self.min = min(self.min, x)

    def __str__(self):
        n = self.n
        sum = self.sum
        sum2 = self.sum2

        assert n > 0
        result = []

        if self.show_total:
            result.append('total={}'.format(float_to_str(sum, 2)))

        result.append('n={}'.format(n))

        mean = 1.0*sum/n
        if n == 1:
            result.append('mean={}'.format(mean))
        else:
            sigma2 = (sum2 - 2*mean*sum + mean*mean*n) / (n-1)
            sigma = sqrt(sigma2)
            result.append(
                'mean={}+-{}'
                .format(float_to_str(mean, 2), float_to_str(sigma/sqrt(n))))
            result.append('sigma={}'.format(float_to_str(sigma)))
            result.append('min={}'.format(self.min))
            result.append('max={}'.format(self.max))

        return ', '.join(result)


times = defaultdict(lambda: Stat(show_total=True))

class TimeIt(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = default_timer()

    def __exit__(self, *args):
        t = default_timer() - self.start
        times[self.name].add_value(float(t))

    def __call__(self, f):
        @wraps(f)
        def timed_f(*args, **kwargs):
            with TimeIt(self.name):
                return f(*args, **kwargs)
        return timed_f


stats = {}

def add_value(name, value):
    if isinstance(value, float):
        stat_class = Stat
    else:
        stat_class = CounterStat

    if name in stats:
        assert isinstance(stats[name], stat_class)
    else:
        stats[name] = stat_class()

    stats[name].add_value(value)


def log_stats():
    logger.info('-'*20)
    for k, v in sorted(stats.items()):
        logger.info('{}: {}'.format(k, v))
    for k, v in sorted(times.items()):
        logger.info('{}: {}'.format(k, v))
    logger.info('-'*20)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    add_value('zzz', 1)
    add_value('x', 2.4)
    add_value('x', 1.5)

    for _ in range(2):
        with TimeIt('qqq'):
            for i in range(10):
                add_value('i', i)

    log_stats()
