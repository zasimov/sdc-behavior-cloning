"""Convert driving_log to Octave Matrix

Usage:

  python -m drivinglog.octave -i driving_log.csv -o driving_log.mat

"""

import argparse
import scipy.io
import os


def die(message):
    print(message)
    exit(1)


if __name__ == '__main__':
    import drivinglog

    parser = argparse.ArgumentParser('convert driving log to Octave matrix')
    parser.add_argument('-i', '--input', required=True, help='an input file (driving_log.csv)')
    parser.add_argument('-o', '--output', required=True, help='an output file')

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        die('input file doesn\'t exist')

    log = drivinglog.read_array(args.input)

    mat = dict(drivinglog=log)

    scipy.io.savemat(args.output, mat)

    exit(0)
