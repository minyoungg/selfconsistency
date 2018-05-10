import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def get_variables(finetune_ckpt_path, exclude_scopes=None):
    """Returns list of variables without scopes that start with exclude_scopes ."""
    if exclude_scopes is not None:
        exclusions = [scope.strip() for scope in exclude_scopes]
        variables_to_restore = [ var for var in slim.get_model_variables() if not np.any([var.op.name.startswith(ex) for ex in exclusions])]
    else:
        variables_to_restore = [ var for var in slim.get_model_variables()]
    return variables_to_restore

def config(use_gpu=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    if use_gpu:
        if type(use_gpu) is list:
            use_gpu = ','.join([str(g) for g in use_gpu])
        config.gpu_options.visible_device_list = str(use_gpu)
    return config

def tfprint(x):
    print x
    return x

def extract_var(starts_with, is_not=False):
    if type(starts_with) is str:
        starts_with = [starts_with]
    selected_vars = []
    for s in starts_with:
        if not is_not:
            selected_vars.extend([var for var in tf.trainable_variables() if var.op.name.startswith(s)])
        else:
            selected_vars.extend([var for var in tf.trainable_variables() if not var.op.name.startswith(s)])
    return selected_vars

def init_solver(param):
    """ Initializes solver using solver param """
    return param.solver(learning_rate=param.learning_rate,
                        beta1=param.beta1,
                        beta2=param.beta2)

def multiclass_accuracy(pr, gt):
    """ pr is logits. computes multiclass accuracy """
    correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(pr)), tf.round(gt))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def leaky_relu(input, slope=0.2):
    """ Leaky relu """
    with tf.name_scope('leaky_relu'):
        return tf.maximum(slope*input, input)

def batch_norm(input, is_training):
    """ batch normalization """
    with tf.variable_scope('batch_norm'):
        return tf.contrib.layers.batch_norm(input, decay=0.9, scale=True,
                            updates_collections=None, is_training=is_training)

def renorm(input, is_training):
    return tf.layers.batch_normalization(input, training=is_training, renorm_momentum=0.9)

def instance_norm(input, is_training):
    """ instance normalization """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable('scale', [num_out],
                initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [num_out],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset

def fc(input, output, reuse=False, norm=None, activation=tf.nn.relu, dropout=0.7, is_training=True, name='fc'):
    """ FC with norm, activation, dropout support """
    with tf.variable_scope(name, reuse=reuse):
        x = slim.fully_connected(input, output, activation_fn=activation, normalizer_fn=norm, reuse=reuse)
        x = tf.nn.dropout(x, dropout)
    return x

def conv(input, output, size, stride,
         reuse=False,
         norm=instance_norm,
         activation=leaky_relu,
         dropout=1.0,
         padding='VALID',
         pad_size=None,
         is_training=True,
         name='conv'):
    """
    Performs convolution -> batchnorm -> relu
    """
    with tf.variable_scope(name, reuse=reuse):
        dropout = 1.0 if dropout is None else dropout
        # Pre pad the input feature map
        x = pad(input, pad_size)
        # Apply convolution
        x = slim.conv2d(x, output, size, stride,
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        padding=padding)
        # Apply dropout
        x = tf.nn.dropout(x, dropout)
        # Apply activation
        x = activation(x) if activation else x
        # Apply normalization
        x = norm(x, is_training) if norm else x
    return x

def pad(input, pad_size):
    """ Reflect pads input by adding pad_size to h x w dimensions """
    if not pad_size:
        return input
    return tf.pad(input, [[0,0],[pad_size, pad_size],[pad_size, pad_size],[0,0]], 'REFLECT')

def average_gradients(grad_list):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    grad_list: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grad_list):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
