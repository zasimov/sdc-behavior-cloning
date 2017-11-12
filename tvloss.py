import argparse
import csv

from matplotlib import pyplot


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python tvloss.py')
    parser.add_argument('--history', required=True, help='a history file')
    parser.add_argument('--output', required=False, help='an output file')

    args = parser.parse_args()

    train = []
    valid = []

    with open(args.history) as f:
        reader = csv.DictReader(f=f)

        for row in reader:
            train.append(float(row['train']))
            valid.append(float(row['valid']))

    pyplot.plot(train)
    pyplot.plot(valid)
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['train', 'valid'])

    if args.output:
        pyplot.savefig(args.output)
    else:
        pyplot.show()
