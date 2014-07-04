import logging
import subprocess
import sys
import time
import traceback
import os
import json
import collections


DIR = 'runs'
MAX_DIFF_SIZE = 10000


class RunRecorder(object):
    def __init__(self):
        self.attrs = collections.OrderedDict()
        self.results = []

        self.attrs['argv'] = sys.argv

        branches = subprocess.check_output('git branch', shell=True)
        for line in branches.splitlines():
            if line.startswith('* '):
                self.attrs['branch'] = line[2:]

        self.attrs['commit'] = subprocess.check_output(
            'git rev-parse HEAD', shell=True).strip()

        log = subprocess.check_output('git log --format=oneline', shell=True)
        self.attrs['commit_number'] = len(log.splitlines())

        # to refresh diff cache
        subprocess.check_output('git status', shell=True)

        diff_stat = subprocess.check_output('git diff HEAD --stat', shell=True)
        if diff_stat.strip():
            self.attrs['clean'] = False
            self.attrs['diff_stat'] = diff_stat
            diff = subprocess.check_output('git diff HEAD', shell=True)
            if len(diff) <= MAX_DIFF_SIZE:
                self.attrs['diff'] = diff
        else:
            self.attrs['clean'] = True

        self.attrs['num_runs'] = 0

        t = self.attrs['start_time'] = time.time()
        self.id = '{:x}'.format(int(t))
        self.filename = os.path.join(DIR, self.id)

    def save(self):
        with open(self.filename, 'w') as fout:
            fout.write('[')
            json.dump(self.attrs, fout, indent=4)
            fout.write(',\n')
            json.dump(self.results, fout)
            fout.write(']')

    def add_result(self, result):
        self.attrs['num_runs'] += 1
        self.results.append(result)

    def __enter__(self):
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        assert not os.path.exists(self.filename)
        self.save()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.attrs['end_time'] = time.time()

        if exc_type is not None:
            self.attrs['error'] = ''.join(
                traceback.format_exception(exc_type, exc_value, tb))

        self.save()


class Run(object):
    def __init__(self, id):
        self.id = id
        self.filename = os.path.join(DIR, id)
        with open(self.filename) as fin:
            self.attrs, self.results = json.load(fin)


def get_all_runs():
    for filename in sorted(os.listdir(DIR)):
        yield Run(filename)


if __name__ == '__main__':
    with RunRecorder() as run:
        run.add_result(dict(score=10))
        run.add_result(dict(score=-1))

    print list(get_all_runs())
