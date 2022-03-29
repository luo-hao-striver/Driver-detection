#!/bin/bash

#模型可选择 'VGG11'| 'VGG13' | 'Resnet18'|'MobileNet'
MODE='MobileNet'

#batch_size
BATCH_SIZE=8

#epoch
EPOCH=15

python ./main.py test --model_type=$MODE --batch_size=$BATCH_SIZE --epoch=$EPOCH
