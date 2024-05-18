import numpy as np
import os
import sys
import random
import tensorflow as tf
import custom_models


def getKeys(*entries:tuple):
    return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

keys = []

mode = 'train'

s_field = list(range(1,41))
g_field = list(range(1,50))
if mode == 'train':
    r_field = [1,3,4,6]
else:
    r_field = [2,5]

# a = tf.constant([[1,2,3,4],[5,6,7,8]])
# a = tf.expand_dims(a, axis=2)


nn = custom_models.simple_conv_net()

print(nn.summary())

print()
#getKeys((1,2,3),(4,5,6))


