#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Dec-01-22 17:37
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

"""
Remember to set: 
export PYTHONPATH="${PYTHONPATH}:./"
"""

from models.resnet_cifar10 import create_resnet_cifar10

create_resnet_cifar10([32, 32, 3], 20, version=1, use_addernet=True)
