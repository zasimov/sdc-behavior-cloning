"""Restore trajectory by driving log
"""

import argparse
import math

import numpy


RAD_DEGREE = math.pi / 180.0


def rad(degree):
    return degree * RAD_DEGREE


class Vector2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def polar(cls, alpha_degree, magnitude):
        alpha_rad = rad(alpha_degree)
        x = magnitude * math.cos(alpha_rad)
        y = magnitude * math.sin(alpha_rad)
        return cls(x, y)

    def add(self, vector):
        return Vector2D(self.x + vector.x, self.y + vector.y)

    def __repr__(self):
        return 'Vector2D(%s, %s)' % (self.x, self.y)

    def to_array(self):
        coords = (self.x, self.y)
        return numpy.array(coords)


def restore(driving_log, frame_rate=10.0, base_angle=90.0, angle=25):
    """restore trajectory using driving_log

    driving_log is a list of entries (see drivinglog.read function)

    restore is a generator, generates Vector2D
    """
    v = Vector2D(0.0, 0.0)

    yield v

    for entry in driving_log:
        alpha_degree = base_angle + angle * entry.steeringangle
        magnitude = round(entry.speed, 2) / frame_rate
        v2 = Vector2D.polar(alpha_degree, magnitude)
        v = v.add(v2)
        yield v


if __name__ == '__main__':
    import drivinglog

    parser = argparse.ArgumentParser('restore trajectory by driving log')
    parser.add_argument('-i', '--input', required=True, help='an input file (driving log)')
    parser.add_argument('-o', '--output', required=True, help='an output file')
    parser.add_argument('--max-speed', default=30.91, type=float)
    parser.add_argument('--frame-rate', default=10, type=float)
    parser.add_argument('--base-angle', default=0, type=float)
    parser.add_argument('--angle', default=25, type=float)

    args = parser.parse_args()

    driving_log = drivinglog.read(args.input)

    with open(args.output, 'w') as out:
        for v in restore(driving_log, frame_rate=args.frame_rate,
                         base_angle=args.base_angle, angle=args.angle):
            out.write('%s %s\n' % (v.x, v.y))


