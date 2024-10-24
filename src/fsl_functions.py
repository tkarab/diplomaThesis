import tensorflow as tf
import numpy as np


"""
retu
"""
def produce_prototype(x):
    return tf.reduce_mean(x,axis=-2)

def reshape_query(x):
    return tf.reshape(x,[-1,tf.shape(x)[-1]])

def keep_first_tensor(x):
    print_tensor_shape(x)
    x = tf.expand_dims(x[0],axis=0)
    print_tensor_shape(x)

    return x

def get_sum_of_squares(x):
    return tf.reduce_sum(x**2,axis=1,keepdims=True)

def euc_dist(args):
    prototypes, query_feat = args

    return tf.sqrt(tf.reduce_sum(tf.square(prototypes-query_feat), axis=-1))

def l2_dist(args):
    x1, x2 = args
    return tf.square(x1-x2)

def l1_dist(args):
    x1, x2 = args
    return tf.abs(x1-x2)

def inner_product(args):
    x1, x2 = args
    return x1*x2

def inner_product_norm(args):
    x1, x2 = args
    return (x1/tf.norm(x1))*(x2/tf.norm(x2))

def inner_product_norm_res(args):
    x1, x2 = args
    y = x1*x2
    # if tf.norm(y) == 0.0:
    #     return tf.zeros_like(y)
    return y/tf.norm(y)

def inner_product_abs(args):
    x1, x2 = args
    return tf.abs(x1*x2)

def inner_product_norm_abs(args):
    x1, x2 = args
    return tf.abs((x1/tf.norm(x1))*(x2/tf.norm(x2)))


def softmax_classification(args, print_result=False):
    pred = tf.nn.softmax(-euc_dist(args),axis=-1)
    if print_result:
        print_tensor(pred)
    return pred

def get_fsl_set_rand(N,k,dim1=28,dim2=28):
    x = tf.random.uniform(shape=(N, k, dim1, dim2, 1), minval=0, maxval=1)
    return x

def get_fsl_set(x,N,k):
    return tf.stack([tf.concat([i*x+j for j in range(k)],axis=0) for i in range(1,N+1)],axis=0)


def print_tensor(tensor, precision=4):
    tf.print(tf.strings.as_string(tensor,precision=precision))

def print_tensor_shape(tensor):
    tf.print("shape:",tensor.shape)


