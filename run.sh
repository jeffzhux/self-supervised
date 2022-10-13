#!/bin/bash

source /home/hsiang/Hsiang/Code/self-supervised/.venv/bin/activate

python main_test.py ./config/cifar10/oursimclr_config.py
python main_test.py ./config/cifar10/simclr_config.py
