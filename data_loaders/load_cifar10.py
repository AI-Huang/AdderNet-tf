#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 15:02
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://keras.io/zh/examples/cifar10_resnet/

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from data_loaders.augmentations import pad_and_crop
from data_loaders.data_sequences import CIFAR10Sequence


def color_normalize(train_images, test_images):

    mean = [np.mean(train_images[:, :, :, i])
            for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i])
           for i in range(3)]  # [62.9932, 62.0887, 66.7048]

    for i in range(3):
        train_images[:, :, :, i] = \
            (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = \
            (test_images[:, :, :, i] - mean[i]) / std[i]

    return train_images, test_images


def load_cifar10(normalize=False,
                 subtract_pixel_mean=False,
                 featurewise_std_normalization=False,
                 validation_split=0.0,
                 seed=None,
                 do_pad_and_crop_once=False,
                 to_categorical=True):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Rescale the data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Normalize the data first
    if normalize:
        x_train, x_test = color_normalize(x_train, x_test)

    x_train_mean = np.mean(x_train, axis=0)  # Per pixel mean?
    x_train_std = np.std(x_train, axis=0)

    # If apply the mean and standard deviation normalisation
    if featurewise_std_normalization:
        subtract_pixel_mean = True

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train -= x_train_mean
        x_test -= x_train_mean

    if featurewise_std_normalization:
        print("featurewise_std_normalization")
        x_train /= x_train_std
        x_test /= x_train_std

    # Convert class vectors to binary class matrices.
    if to_categorical:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)

    # Train val split
    x_val, y_val = None, None
    if validation_split > 0:
        print(f"Using validation_split: {validation_split}.")
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=validation_split, random_state=seed)

    # Only pad and crop on the train set
    if do_pad_and_crop_once:
        print(f"Using TensorFlow augmentation to do pre-processing once: pad_and_crop.")
        x_train = pad_and_crop(x_train)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar10_sequence(batch_size=128,
                          shuffle=True,
                          seed=42,
                          normalize=False,
                          subtract_pixel_mean=True,
                          validation_split=0,
                          to_categorical=True,
                          data_augmentation=False):

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10(normalize=normalize,
                                                                        subtract_pixel_mean=subtract_pixel_mean,
                                                                        to_categorical=to_categorical)

    if data_augmentation:
        transforms = [pad_and_crop]
    else:
        transforms = None

    cifar10_sequence_train = CIFAR10Sequence(x_train, y_train,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             seed=seed,
                                             subset="training",
                                             validation_split=validation_split,
                                             transforms=transforms)

    cifar10_sequence_val = CIFAR10Sequence(x_train, y_train,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           seed=seed,
                                           subset="validation",
                                           validation_split=validation_split,
                                           transforms=transforms)

    cifar10_sequence_test = CIFAR10Sequence(x_test, y_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            subset="validation",
                                            validation_split=1.0,
                                            transforms=None)

    return cifar10_sequence_train, cifar10_sequence_val, cifar10_sequence_test


def main():
    cifar10_sequence_train, cifar10_sequence_val, cifar10_sequence_test = \
        load_cifar10_sequence(batch_size=128,
                              shuffle=True,
                              seed=42,
                              norm=False,
                              subtract_pixel_mean=True,
                              validation_split=0.1,
                              to_categorical=True)


if __name__ == "__main__":
    main()
