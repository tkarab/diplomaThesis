import numpy as np
import os
import sys
import random
import tensorflow as tf
import fsl_functions
# import custom_models
import constants
import time
import helper_functions
import plot_functions

def getKeys(*entries:tuple):
    return ['s{}g{}r{}'.format(s,g,r) for s,r,g in entries]

DATABASE = 2


support = tf.random.uniform(minval=0, maxval=10, shape = (2,3,5), dtype=tf.int32)
query = tf.random.uniform(minval=0, maxval=10, shape = (2,1,5), dtype=tf.int32)
dif = support - query
dif_sq = tf.square(dif)
dif_sq_sum = tf.reduce_sum(dif_sq,axis=-1)
dif_sqrt = tf.sqrt(tf.cast(dif_sq_sum,dtype=tf.float32))
output = tf.nn.softmax(-dif_sqrt)
print()

