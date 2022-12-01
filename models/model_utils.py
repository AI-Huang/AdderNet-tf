#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-27-20 21:46
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""Model configuration functions
- create_model
- create_optimizer
- set_metrics

Requirements:
    Kan HUANG's models repo: https://github.com/AI-Huang/models
"""

import tensorflow as tf
import tensorflow_addons as tfa
# from models.tf_fn.lenet import LeNet5, LeCunLeNet5
# ResNet for CIFAR-10
from models.addernet_cifar10 import resnet_v1, resnet_v2


def create_model(model_name, **kwargs):
    """Create model with model's model_name
    """
    assert "input_shape" in kwargs
    assert "num_classes" in kwargs
    input_shape = kwargs["input_shape"]
    num_classes = kwargs["num_classes"]

    """
    if model_name == "LeNet5":
        model = LeNet5(input_shape=input_shape, num_classes=num_classes)

    elif model_name == "LeCunLeNet5":
        model = LeCunLeNet5(input_shape=input_shape, num_classes=num_classes)

    elif model_name.startswith("AttentionLeNet5"):
        from .attention_lenet import AttentionLeNet5
        model = AttentionLeNet5(input_shape=input_shape,
                                num_classes=num_classes,
                                attention="senet")

    elif model_name == "ResNet18":
        from models.keras_fn.resnet_extension import ResNet18
        model = ResNet18(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif model_name == "ResNet34":
        from models.keras_fn.resnet_extension import ResNet34
        model = ResNet34(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )
    """

    if model_name == "ResNet50":
        from tensorflow.keras.applications.resnet import ResNet50
        model = ResNet50(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif model_name == "ResNet101":
        from tensorflow.keras.applications.resnet import ResNet101
        model = ResNet101(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    elif model_name == "ResNet152":
        from tensorflow.keras.applications.resnet import ResNet152
        model = ResNet152(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )

    else:
        raise Exception('Unknown model: ' + model_name)

    """
    elif model_name == "ResNet20v2":  # "ResNet20v2",  "ResNet56v2"
        # hparams: n, version, input_shape, num_classes
        assert "n" in kwargs
        assert "version" in kwargs
        n = kwargs["n"]
        version = kwargs["version"]

        from .fault_resnet import model_depth, resnet_v2, lr_schedule
        depth = model_depth(n=2, version=2)
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
        # TODO
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))
    """

    return model


def create_model_cifar10(depth, se_net=False, version=1):
    input_shape = [32, 32, 3]
    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, se_net=se_net, depth=depth)
    return model


def create_optimizer(optimizer_name="Adam", **kwargs):

    if optimizer_name == "Adam":
        # Default values
        learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs else 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif optimizer_name == "SGD":
        learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs else 0.0
        weight_decay = kwargs["weight_decay"] if "weight_decay" in kwargs else None
        momentum = kwargs["momentum"] if "momentum" in kwargs else 0.0
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            # weight_decay=weight_decay, # Unavailable for tf2.5
            nesterov=False, name='SGD')

    elif optimizer_name == "SGDW":
        learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs else 0.001
        weight_decay = kwargs["weight_decay"] if "weight_decay" in kwargs else 0.0001
        momentum = kwargs["momentum"] if "momentum" in kwargs else 0.0
        optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum)

    else:
        raise Exception("Unknown optimizer: " + optimizer_name)

    return optimizer
