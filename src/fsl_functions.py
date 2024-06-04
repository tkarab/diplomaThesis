import tensorflow as tf


"""
retu
"""
def produce_prototype(x):
    return tf.reduce_mean(x,axis=1)

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

    return tf.sqrt(tf.reduce_sum(tf.square(prototypes-query_feat), axis=1))

def l2_dist(args):
    prototypes, query_feat_reshaped = args
    p_exp = tf.expand_dims(prototypes,axis=0)
    q_exp = tf.expand_dims(query_feat_reshaped, axis=1)
    diff = q_exp - p_exp
    return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))

def softmax_classification(args):
    pred = tf.nn.softmax(-euc_dist(args))
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


