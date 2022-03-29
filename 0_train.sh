#!/bin/bash

#模型可选择 'VGG11'| 'VGG13' | 'Resnet18' | 'MobileNet'
MODE='Resnet18'

#batch_size
BATCH_SIZE=8

#epoch
EPOCH=10

python ./main.py train  --model_type=$MODE --batch_size=$BATCH_SIZE --epoch=$EPOCH


