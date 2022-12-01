#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-09-21 20:02
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import tensorflow as tf


def cifar10_scheduler(epoch):
    """Learning rate schedule for training ResNet on CIFAR-10

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs that model has been trained for

    # Returns
        lr (float32): learning rate
    """

    lr = 0.1  # base learning rate
    if 92 <= epoch < 137:
        lr *= 0.1  # reduce factor
    elif 137 <= epoch < 182:
        lr *= 0.01
    elif epoch >= 182:
        lr *= 0.001

    print(
        f"Model has been trained for {epoch} epoch(s); learning rate for next epoch: {lr}.")

    tf.summary.scalar('learning rate', data=lr, step=epoch)

    return lr


def keras_lr_scheduler(epoch):
    """Keras's polynomial learning rate schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs that model has been trained for

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3  # base learning rate
    if 80 <= epoch < 120:
        lr *= 1e-1  # reduce factor
    elif 120 <= epoch < 160:
        lr *= 1e-2
    elif 160 <= epoch < 180:
        lr *= 1e-3
    elif epoch >= 180:
        lr *= 0.5e-3

    print(
        f"Model has been trained for {epoch} epoch(s); learning rate for next epoch: {lr}.")

    tf.summary.scalar('learning rate', data=lr, step=epoch)

    return lr
