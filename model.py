"""NVidia Dave model trainer
"""

import argparse
import csv

import udacitylib
import drivinglog

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers import Dropout

from sklearn.utils import shuffle


def save_history(history, file_name):
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    with open(file_name, 'w') as out:
        writer = csv.DictWriter(f=out, fieldnames=['train', 'valid'])

        writer.writeheader()

        for train, valid in zip(train_loss, valid_loss):
            row = dict(
                train=train,
                valid=valid,
            )
            writer.writerow(row)


def infinite(stream, batch_size):
    while True:
        batches = stream(batch_size)
        for batch in batches:
            yield shuffle(*batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python model.py')
    parser.add_argument('--dataset', required=True, help='a dataset (file name)')
    parser.add_argument('--model', required=True, help='an output model (file name)')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--history', required=False, help='a file name of history file (csv)')

    args = parser.parse_args()

    # Build model
    model = Sequential ()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mse', optimizer='adam')

    # Train
    with udacitylib.HDF5Samples(args.dataset) as dataset:
        train = dataset.group('train',
                              targets_fn=lambda ts: ts[:, drivinglog.NP_FIELD_STEERINGANGLE])
        # valid target is a 4-vector
        # we use only steering angle (see targets_fn below)
        valid = dataset.group('valid',
                              targets_fn=lambda ts: ts[:, drivinglog.NP_FIELD_STEERINGANGLE])


        train_generator = infinite(train.raw_batches, args.batch_size)
        valid_generator = infinite(valid.raw_batches, args.batch_size)

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train) / args.batch_size,
            validation_data=valid_generator,
            validation_steps=len(valid) / args.batch_size,
            nb_epoch=args.epochs,
        )

        model.save(args.model)

    # Save training history
    if args.history:
        save_history(history, args.history)
