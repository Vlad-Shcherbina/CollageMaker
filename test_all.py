import sys

import nose


if __name__ == '__main__':
    nose.run_exit(argv=sys.argv + [
        '--verbose', '--with-doctest',
        '--logging-level=DEBUG'
        ])
