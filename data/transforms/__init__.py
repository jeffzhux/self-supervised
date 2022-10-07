"""The lightly.transforms package provides additional augmentations.
    Contains implementations of Gaussian blur and random rotations which are
    not part of torchvisions transforms.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from data.transforms.gaussian_blur import GaussianBlur
from data.transforms.rotation import RandomRotate
from data.transforms.solarize import RandomSolarization
from data.transforms.jigsaw import Jigsaw
from data.transforms.transform import cifar_linear
from data.transforms.transform import cifar_test
from data.transforms.transform import SimCLRTransform