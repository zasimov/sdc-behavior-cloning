"""drivinglog package implements functions to work with Udacity Driving Log

drivinglog uses 'csv' module to parse driving_log.csv

The standard way to use this module:

  data = [entry for entry in drivinglog.read('driving_log.csv')]

"""

import collections
import csv
import os
import typing

import cv2
import numpy

# Field numbers
# You can use this numbers to read values directly from row.
# However, it's better to use 'read' generator.
FIELD_CENTERCAM = 0
FIELD_LEFTCAM = 1
FIELD_RIGHTCAM = 2
FIELD_STEERINGANGLE = 3
FIELD_THROTTHLE = 4
FIELD_BREAK = 5
FIELD_SPEED = 6


# These values for the resulf of 'read_array' function
NP_FIELD_STEERINGANGLE = FIELD_STEERINGANGLE - 3
NP_FIELD_THROTTHLE = FIELD_THROTTHLE - 3
NP_FIELD_BREAK = FIELD_BREAK - 3
NP_FIELD_SPEED = FIELD_SPEED - 3


# NOTE: there is breaklevel, no break, because break is reserved word
Entry = collections.namedtuple('Entry', [
    'centercam', 'leftcam', 'rightcam',
    'steeringangle', 'throttle', 'breaklevel', 'speed',
    ])


def read(filename) -> typing.List[Entry]:
    """read reads entries from drivelog

    read is a generator
    """
    with open(filename, 'r') as driving_log:
        reader = csv.reader(driving_log)
        for row in reader:
            for i in range(FIELD_STEERINGANGLE, FIELD_SPEED + 1):
                row[i] = float(row[i])
            yield Entry(*row)


def read_array(filename) -> numpy.ndarray:
    """like a 'read' but returns numpy array (floats only)
    """
    def section(entry: Entry):
        floats = (entry.steeringangle, entry.throtthle, entry.breakelevel, entry.speed)
        return numpy.array(floats)

    floats_only = [section(entry) for entry in read(filename)]
    return numpy.array(floats_only, dtype=numpy.float)


def limit(max_abs, sa):
    if sa > 0:
        return min(max_abs, sa)
    return max(-max_abs, sa)


class DrivingLogEntry:

    def __init__(self, driving_log, entry):
        self.driving_log = driving_log
        self.entry = entry
        self._leftcam = None
        self._centercam = None
        self._rightcam = None

    def _read_image(self, filename):
        """_read_image reads an image from file in BGR
        
        Note: cv2 reads "transposed" image by default (first dimension is height).
        """
        return cv2.imread(filename)

    def _fixed_path(self, image_path):
        file_name = os.path.basename(image_path)
        return os.path.join(self.driving_log.img_path, file_name)

    @property
    def leftcam(self):
        if self._leftcam is None:
            self._leftcam = self._read_image(self._fixed_path(self.entry.leftcam))
        return self._leftcam

    @property
    def centercam(self):
        if self._centercam is None:
            self._centercam = self._read_image(self._fixed_path(self.entry.centercam))
        return self._centercam

    @property
    def rightcam(self):
        if self._rightcam is None:
            self._rightcam = self._read_image(self._fixed_path(self.entry.rightcam))
        return self._rightcam

    @property
    def steeringangle(self):
        return self.entry.steeringangle

    @property
    def throttle(self):
        return self.entry.throttle

    @property
    def breaklevel(self):
        return self.entry.breaklevel

    @property
    def speed(self):
        return self.entry.speed

    @property
    def targets(self):
        return (self.steeringangle, self.throttle, self.breaklevel, self.speed)

    @property
    def flipped_targets(self):
        return (-self.steeringangle, self.throttle, self.breaklevel, self.speed)

    def corrected_targets(self, steering_add, max_abs=1.0):
        new_sa = self.steeringangle + steering_add
        new_sa = limit(max_abs, new_sa)
        return (new_sa, self.throttle, self.breaklevel, self.speed)


class DrivingLog:
    """DrivingLog represents folder with driving data"""

    # a name of IMG folder (relative)
    IMG_FOLDER_NAME = 'IMG'

    # a file name of driving log
    DRIVING_LOG_CSV = 'driving_log.csv'

    DRIVING_LOG_ENTRY_CLS = DrivingLogEntry

    def __init__(self, folder_path):
        self.path = folder_path
        if not os.path.isdir(self.path):
            raise ValueError('a folder with driving log doesn\'t exist')
        self._entries = None

    @property
    def img_path(self):
        return os.path.join(self.path, self.IMG_FOLDER_NAME)

    @property
    def driving_log_csv(self):
        return os.path.join(self.path, self.DRIVING_LOG_CSV)

    @property
    def entries(self):
        if not self._entries:
            self._entries = [self.DRIVING_LOG_ENTRY_CLS(self, entry) for entry in read(self.driving_log_csv)]
        return self._entries
