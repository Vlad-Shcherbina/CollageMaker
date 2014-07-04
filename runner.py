import os
import re
import subprocess
from timeit import default_timer
import multiprocessing
import math


def run_solution(seed):
    try:
        start = default_timer()
        p = subprocess.Popen(
            'cd data; '
            'java -jar CollageMakerVis.jar -exec "python ../sol.py" '
            '-novis -seed {}'.format(seed),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out, err = p.communicate()
        p.wait()
        assert p.returncode == 0

        result = {}
        for line in out.splitlines():
            var, _, value = line.strip().partition(' = ')
            if value:
                result[var.strip()] = eval(value)

        result['seed'] = seed
        result['time'] = default_timer() - start

        assert result['Score'] > 0
        result['log_score'] = -math.log(result['Score'])

        return result
    except Exception as e:
        print e
        raise Exception('seed={}, out={}, err={}'.format(seed, out, err))


if __name__ == '__main__':
    import run_db

    seeds = [i*1000 + j for i in range(5) for j in range(1, 11)]

    map = multiprocessing.Pool(5).imap

    with run_db.RunRecorder() as run:
        for result in map(run_solution, seeds):
            print result['seed'], result['log_score']
            run.add_result(result)
            run.save()

    #print run_solution(1)
