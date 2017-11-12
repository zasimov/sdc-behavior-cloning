"""sahist calculates histogram of steering angle
"""

import drivinglog
import udacitylib

from matplotlib import pyplot


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='HDF5 file name')
    parser.add_argument('--bins', default=20, type=int, help='number of bins')
    parser.add_argument('--output', required=False, help='an output file')

    args = parser.parse_args()

    with udacitylib.HDF5Samples(args.dataset) as ds:
        train = ds.group('train')
        valid = ds.group('valid')

        sa = train.targets[:, drivinglog.NP_FIELD_STEERINGANGLE]
        pyplot.hist(sa, bins=args.bins)

        sa = valid.targets[:, drivinglog.NP_FIELD_STEERINGANGLE]
        pyplot.hist(sa, bins=args.bins)

        pyplot.xlabel('steering angle')
        pyplot.ylabel('count')
        pyplot.legend(['train (%s)' % len(train), 'valid (%s)' % len(valid)])

    if args.output:
        pyplot.savefig(args.output)
    else:
        pyplot.show()

