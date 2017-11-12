"""Convert driving logs to single HDF5 file 

NOTE: Current implementation requires A LOT OF RAM (~ 8 Gb)
"""

import logging
import os
import sys

import drivinglog
from drivinglog import samples

import h5py
import numpy
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


logger = logging.getLogger(__name__)


def die(message):
    sys.stderr.write(message)
    sys.stderr.write('\n')
    exit(1)


def dlog(folder_path):
    """dlog creates drivinglog.DrivingLog object"""

    if not os.path.isdir(folder_path):
        die('folder does\'t exist: %s' % folder_path)

    dlog = drivinglog.DrivingLog(folder_path)

    if not os.path.isdir(dlog.img_path):
        die('IMG folder does\'t exist for: %s' % folder_path)

    if not os.path.isfile(dlog.driving_log_csv):
        die('driving_log.csv file does\'t exist for: %s' % folder_path)

    return dlog


def save_as_hdf5(features, targets):
    # TODO: save features and targets to hdf5
    pass


def save_as_mat(features, targets):
    # TODO: save features and targets to mat (Octave, Matlab)
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('python dataset.py')
    parser.add_argument('--drivinglog', nargs='+', metavar='FOLDER', help='a path of folder')
    parser.add_argument('--left-v', default=0.2, type=float)
    parser.add_argument('--right-v', default=-0.2, type=float)
    parser.add_argument('--output', help='an output file (HDF5)')
    parser.add_argument('--targets-dtype', default='float32')
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--random-state', default=42, type=int)
    parser.add_argument('--valid-size', default=0.2, type=float)
    parser.add_argument('--flip-lr', nargs='+', metavar='DRIVINGLOG', help='full path for driving log that should be filpped l-r')

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)8s  %(asctime)s    %(message)s', level=logging.INFO)

    if not args.drivinglog:
        die("there is no driving logs")

    for fn in args.drivinglog:
        logger.info('Use driving log %s', fn)

    # verify driving logs
    # 1. driving log folder exists
    # 2. driving log folder contains driving_log.csv
    # 3. driving log folder contains IMG subfolder
    dlogs = [dlog(folder_path) for folder_path in args.drivinglog]

    # dlogs contains a list of drivinglog.DrivingLog

    if os.path.isfile(args.output):
        os.unlink(args.output)

    flip_lr = args.flip_lr or []
    flip_lr = set(flip_lr)

    out = h5py.File(args.output)

    train_features = []
    train_targets = []
    valid_features = []
    valid_targets = []

    for dlog in dlogs:
        flip_flag = (dlog.path in flip_lr)
        logger.info('Process %s with flip %s', dlog.path, flip_flag)
        features, targets = samples.lcr(dlog, left_v=args.left_v, right_v=args.right_v,
                                        flip_center_lr=flip_flag)

        if args.shuffle:
            logger.info('Shuffle %s', dlog.path)
            features, targets = shuffle(features, targets)

        logger.info('Split %s, valid size is %s', dlog.path, args.valid_size)
        dlog_train_features, dlog_valid_features, dlog_train_targets, dlog_valid_targets = train_test_split(
            features, targets, test_size=args.valid_size, random_state=args.random_state)

        train_features.extend(dlog_train_features)
        train_targets.extend(dlog_train_targets)
        valid_features.extend(dlog_valid_features)
        valid_targets.extend(dlog_valid_targets)

    if args.shuffle:
        logger.info('Shuffle...')
        train_features, train_targets = shuffle(train_features, train_targets)
        valid_features, valid_targets = shuffle(valid_features, valid_targets)

    train = out.create_group('train')
    train.create_dataset('features', data=numpy.array(train_features))
    train.create_dataset('targets', data=numpy.array(train_targets), dtype=args.targets_dtype)

    valid = out.create_group('valid')
    valid.create_dataset('features', data=numpy.array(valid_features))
    valid.create_dataset('targets', data=numpy.array(valid_targets), dtype=args.targets_dtype)

    out.close()

    logger.info('Saved to %s', args.output)
