import sys, os
import init_paths
import tensorflow as tf

def initialize_exif(ckpt='', init=True, use_gpu=0):
    from models.exif import exif_net, exif_solver
    tf.reset_default_graph()
    net_args = {'num_classes':80+3,
                'is_training':False,
                'train_classifcation':True,
                'freeze_base': True,
                'im_size':128,
                'batch_size':64,
                'use_gpu':[use_gpu],
                'use_tf_threading':False,
                'learning_rate':1e-4}

    solver = exif_solver.initialize({'checkpoint':ckpt,
                                     'use_exif_summary':False,
                                     'init_summary':False,
                                     'exp_name':'eval'})
    if init:
        net = exif_net.initialize(net_args)
        solver.setup_net(net=net)
        return solver
    return solver, exif_net, net_args
