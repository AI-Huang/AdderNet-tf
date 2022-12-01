#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-11-22 15:00
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import numpy as np
import tensorflow as tf


class CIFAR10Sequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x,
        y,
        batch_size=32,
        shuffle=False,
        seed=42,
        subset="training",
        # set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set
        validation_split=0,
        transforms=None
    ):
        """Initialize the windowed sub-sequences dataset generated from a sequence data.

        Arguments:
            x: a numpy array, image data, x_train or x_test.
            y: corresponding y to the x data, y_train or y_test.
            batch_size: batch size, default 32.
            shuffle: whether to shuffle the data index initially **and** after each epoch.
            seed: set the initial random seed for the first time of shuffling sample indices when using numpy.random.shuffle().
            subset: one of "training" and "validation", default "training". If the data is split into training and validation sets, the set this data generator belongs to.
            validation_split: ratio of validation set, e.g., set to 0.2, if you want 0.8 of the samples as training set and 0.2 as validation set.
            transforms is a `tf.keras.Sequential` object.
        """

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.transforms = transforms

        self.image_size = x[0].shape
        self.label_size = y[0].shape

        self.index = self._set_index_array()

        self.shuffle = shuffle
        if subset not in ["training", "validation"]:
            raise ValueError(
                """subset must be one of ["training", "validation"]!""")
        self.subset = subset

        # shuffle firstly, if shuffle is True
        if self.shuffle:
            # set random seet for the first time of shuffling
            np.random.seed(seed)
            self.shuffle_index()

        # split secondly
        self.validation_split = validation_split
        self.set_subset()

    def __getitem__(self, batch_index):
        """Gets batch at batch_index `batch_index`.

        Arguments:
            batch_index: batch_index of the batch in the Sequence.

        Returns:
            batch_x, batch_y: a batch of sequence data.
        """
        x, y = self.x, self.y

        batch_size = self.batch_size

        # sample from (x, y) by `sample_indices`
        sample_indices = \
            self.index[batch_index * batch_size:(batch_index+1) * batch_size]
        batch_x = np.empty((batch_size, *self.image_size))
        batch_y = np.empty((batch_size, *self.label_size))
        for _, idx in enumerate(sample_indices):
            batch_x[_, ], batch_y[_, ] = x[idx, ], y[idx, ]

        # Augment the images after fetching them
        if self.transforms is not None:
            for transform in self.transforms:
                batch_x = transform(batch_x)

        return batch_x, batch_y

    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.index.shape[0] / self.batch_size))

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        # shuffle data index on the end of every epoch if self.shuffle is True
        if self.shuffle:
            self.shuffle_index()

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def _set_index_array(self):
        """_set_index_array
        """
        # index of the beginning element of the window samples
        # +1 makes the index space a close interval
        N = self.x.shape[0]
        return np.arange(0, N)

    def set_subset(self):
        """set_subset
        split indices according to the 'validation_split' and 'subset' attribute,
        this code will change the 'index' attribute.
        """
        if 0 <= self.validation_split < 1:
            i = int(self.validation_split * len(self.index))
            # cut ending elements from the ith index
            if self.subset == "training":
                # the ending part is the training set
                self.index = self.index[i:]
            elif self.subset == "validation":
                # the front part is the validation set
                self.index = self.index[:i]

    def shuffle_index(self):
        """shuffle data index
        """
        np.random.shuffle(self.index)
