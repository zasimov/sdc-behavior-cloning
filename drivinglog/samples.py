"""drivinglog.samples contains functions to calculate samples
"""

import numpy


def center(driving_log):
    """center calculates features and targets using centercam only
    
    Returns tuple (features, targets) where
    
      features a list of center images (numpy arrays)
      
      targets a vector of measurements (steering angle, ...)
    """
    features = []
    targets = []
    for entry in driving_log.entries:
        features.append(entry.centercam)
        targets.append(entry.targets)
    return (numpy.array(features), numpy.array(targets))


def lcr(driving_log, left_v=0.2, right_v=-0.2, flip_center_lr=True):
    """lcr calculates features and targets using leftcam, centercam and rightcam
    
    (lcr = left-center-right)
    
    Arguments:
    
      - driving_log is a drivinglog.DrivingLog instance
      
      - left_v (float) - steering angle correction for left cam image
      
      - right_v (float) - steering angle correction for right cam image
      
      - flip_center_lr (bool) - generate more date using numpy.fliplr 
    
    
    Returns tuple (features, targets) where features and targets are python lists.
    
    You can convert features and targets to numpy arrays using numpy.array(lst).
    
    """

    features = []
    targets = []

    for entry in driving_log.entries:
        features.append(entry.leftcam)
        targets.append(entry.corrected_targets(left_v))

        features.append(entry.centercam)
        targets.append(entry.targets)

        if flip_center_lr:
            flipped = numpy.fliplr(entry.centercam)
            features.append(flipped)
            targets.append(entry.flipped_targets)

        features.append(entry.rightcam)
        targets.append(entry.corrected_targets(right_v))

    return (features, targets)
