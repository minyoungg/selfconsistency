import tensorflow as tf
from utils import ops
import copy, numpy as np
from nets import resnet_v2, resnet_utils
slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


class EXIFNet():
    """
    Given a patch from an image try to classify which camera model it came from
    """
    def __init__(self, num_classes=83, train_classifcation=False,
                 use_tf_threading=False, train_runner=None, batch_size=None,
                 im_size=128, is_training=True, freeze_base=False, use_gpu=0,
                 learning_rate=1e-4, use_classify_with_feat=False):
        """
        num_classes: Number of EXIF classes to predict
        classify_with_feat: If True, the classification layers use the output
                            ResNet features along with EXIF predictions
        train_classifcation: Trains a classifer on top of the EXIF predictions
        use_tf_threading: Uses tf threading
        train_runner: The queue_runnner associated with tf threading
        batch_size: Batch size of the input variables. Must be specfied if using
                    use_tf_threading and queue_runner
        im_size: image size to specify for the placeholder.
                 Assumes square input for now.
        is_training: When False, use training statistics for normalization.
                     This can be overwritten by the feed_dict
        use_gpu: List of GPUs to use to train
        freeze_base: Freezes all the layers except the classification.
                     train_classifcation must be set to True to be useful.
                     No loss is computed with self.label
        learning_rate: Learning rate for the optimizer
        """

        self.use_gpu = use_gpu if type(use_gpu) is list else [use_gpu]
        self.im_size = im_size
        self.num_classes = num_classes
        self.use_classify_with_feat = use_classify_with_feat
        self.train_classifcation = train_classifcation
        self.freeze_base = freeze_base
        self._is_training = is_training # default value if not provided
        self.use_tf_threading = use_tf_threading
        self.train_runner = train_runner
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if self.use_tf_threading:
            assert self.batch_size is not None, self.batch_size
            assert self.batch_size % len(use_gpu) == 0, 'batch size should be modulo of the number of gpus'
            im_a, im_b, label = self.train_runner.get_inputs(self.batch_size)
            self.im_a = tf.placeholder_with_default(im_a, [None, self.im_size, self.im_size, 3])
            self.im_b = tf.placeholder_with_default(im_b, [None, self.im_size, self.im_size, 3])
            self.label = tf.placeholder_with_default(label, [None, self.num_classes])
            self.cls_label = tf.placeholder(tf.float32, [None, 1])
        else:
            self.im_a  =  tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.im_b  =  tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 3])
            self.label =  tf.placeholder(tf.float32, [None, self.num_classes])
            self.cls_label = tf.placeholder(tf.float32, [None, 1])

        self.is_training = tf.placeholder_with_default(self._is_training, None)

        self.extract_features = self.extract_features_resnet50
        # if precomputing, need to populate via feed dict then compute with selecting indices
        # self.im_a_ind, self.im_b_ind
        self.precomputed_features = tf.placeholder(tf.float32, [None, 4096])
        # Add second precompute_features_b for different patch gridding
        self.im_a_index = tf.placeholder(tf.int32, [None,])
        self.im_b_index = tf.placeholder(tf.int32, [None,])
                        
                        
        self.pc_im_a_feat = tf.map_fn(self.mapping_fn, self.im_a_index, dtype=tf.float32,
                                 infer_shape=False)   
        self.pc_im_a_feat.set_shape((None, 4096))
        self.pc_im_b_feat = tf.map_fn(self.mapping_fn, self.im_b_index, dtype=tf.float32,
                                 infer_shape=False)  
        self.pc_im_b_feat.set_shape((None, 4096))
        
        self.model()

        self.cls_variables = ops.extract_var(['classify'])
        
        return

    def get_variables(self):
        """
        Returns only variables that are needed. If freeze_base is True, return
        only variables that start with 'classify'
        """
        if self.freeze_base:
            var_list = ops.extract_var('classify')
        else:
            var_list = tf.trainable_variables()

        assert len(var_list) > 0, 'No variables are linked to the optimizer'
        return var_list
    
    def mapping_fn(self, v):
        # v is an index into precompute_features
        return self.precomputed_features[v]

    def model(self, preemptive_reuse=False):
        """
        Initializes model to train.
        Supports multi-GPU.
        Initializes the optimizer in the network graph.
        """
        with tf.variable_scope(tf.get_variable_scope()):
            # Split data into n equal batches
            im_a_list  = tf.split(self.im_a, len(self.use_gpu))
            im_b_list  = tf.split(self.im_b, len(self.use_gpu))
            label_list = tf.split(self.label, len(self.use_gpu))
            if self.train_classifcation:
                cls_label_list = tf.split(self.cls_label, len(self.use_gpu))

            # We intialize the optimizer here
            self._opt =  tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Used to average
            all_grads      = []
            all_out        = []
            all_cls_out    = []
            all_loss       = []
            all_cls_loss   = []
            all_total_loss = []

            for i, gpu_id in enumerate(self.use_gpu):
                print('Initializing graph on gpu %i' % gpu_id)
                with tf.device('/gpu:%d' % gpu_id):
                    if preemptive_reuse:
                        tf.get_variable_scope().reuse_variables()
                        
                    total_loss = 0
                    im_a, im_b, label = im_a_list[i], im_b_list[i], label_list[i]
                    if self.train_classifcation:
                        cls_label = cls_label_list[i]

                    with tf.name_scope('extract_feature_a'):
                        im_a_feat = self.extract_features(im_a, name='feature_resnet')
                        self.im_a_feat = im_a_feat
                        
                    with tf.name_scope('extract_feature_b'):
                        im_b_feat = self.extract_features(im_b, name='feature_resnet', reuse=True)
                        self.im_b_feat = im_b_feat

                    with tf.name_scope('predict_same'):
                        feat_ab = tf.concat([im_a_feat, im_b_feat], axis=-1)
                        out = self.predict(feat_ab, name='predict')
                        all_out.append(out)
                        
                        pc_feat_ab = tf.concat([self.pc_im_a_feat, self.pc_im_b_feat], axis=-1)
                        pc_out = self.predict(pc_feat_ab, name='predict', reuse=True)

                    if not self.freeze_base:
                        with tf.name_scope('exif_loss'):
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=out))
                            all_loss.append(loss)
                            total_loss += loss
                    
                    if self.train_classifcation:
                        with tf.name_scope('predict_same_image'):
                            if self.use_classify_with_feat:
                                cls_out = self.classify_with_feat(im_a_feat, im_b_feat, out, name='classify')
                                pc_cls_out = self.classify_with_feat(pc_im_a_feat, pc_im_b_feat, pc_out, name='classify',
                                                                     reuse=True)
                            else:
                                cls_out = self.classify(out, name='classify')
                                pc_cls_out = self.classify(pc_out, name='classify', reuse=True)
                            all_cls_out.append(cls_out)
                        with tf.name_scope('classification_loss'):
                            cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=cls_label, logits=cls_out))
                            all_cls_loss.append(cls_loss)
                            total_loss += cls_loss

                    tf.get_variable_scope().reuse_variables()
                    grad = self._opt.compute_gradients(total_loss, var_list=self.get_variables())
                    all_grads.append(grad)
                    all_total_loss.append(total_loss)

        # Average the gradient and apply
        avg_grads = ops.average_gradients(all_grads)
        self.all_loss = all_loss
        self.avg_grads = avg_grads

        if not self.freeze_base:
            self.loss = tf.reduce_mean(all_loss)

        if self.train_classifcation:
            self.cls_loss = tf.reduce_mean(all_cls_loss)

        self.total_loss = tf.reduce_mean(all_total_loss)
        self.opt  = self._opt.apply_gradients(avg_grads) # trains all variables for now

        # For logging results
        self.out     = tf.concat(all_out, axis=0)
        self.pred    = tf.sigmoid(self.out)

        if not self.freeze_base:
            correct_prediction = tf.equal(tf.round(self.pred), tf.round(self.label))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if self.train_classifcation:
            self.cls_out = tf.concat(all_cls_out, axis=0)
            self.cls_pred = tf.sigmoid(self.cls_out)
            self.pc_cls_pred = tf.sigmoid(pc_cls_out)
            self.cls_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.cls_pred), tf.round(self.cls_label)), tf.float32))
        return

    def extract_features_resnet50(self, im, name, is_training=True, reuse=False):
        use_global_pool = True
        num_classes = 4096 if use_global_pool else 512
        with tf.name_scope(name):
            with slim.arg_scope(resnet_utils.resnet_arg_scope()):
                out, _ = resnet_v2.resnet_v2_50(inputs=im,
                                                num_classes=num_classes,
                                                global_pool=use_global_pool,
                                                is_training=self.is_training,
                                                spatial_squeeze=True,
                                                scope='resnet_v2_50',
                                                reuse=reuse)

                if not use_global_pool:
                    args = {'reuse':reuse, 'norm':None, 'activation':tf.nn.relu ,'padding':'SAME', 'is_training':is_training}
                    out_args = copy.deepcopy(args)
                    out_args['activation'] = None
                    out = ops.conv(out, 1024, 3, 2, name='conv1', **args)
                    out = slim.batch_norm(out)
                    out = ops.conv(out, 2048, 3, 2, name='conv2', **args)
                    out = slim.batch_norm(out)
                    out = ops.conv(out, 4096, 3, 2, name='conv3', **out_args)
                    out = slim.batch_norm(out)
                    out = tf.squeeze(out, [1, 2], name='SpatialSqueeze')
        return out

    def predict(self, feat_ab, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            in_size = int(feat_ab.get_shape()[1])
            out = slim.stack(feat_ab, slim.fully_connected, [4096, 2048, 1024], scope='fc')
            out = slim.fully_connected(out, self.num_classes, activation_fn=None, scope='fc_out')
        return out

    def classify_with_feat(self, im_a_feat, im_b_feat, affinity_pred, name, is_training=True, reuse=False):
        """ Predicts whether the 2 image patches are from the same image """
        with tf.variable_scope(name, reuse=reuse):
            x = tf.concat([im_a_feat, im_b_feat, affinity_pred], axis=-1)
            x = slim.stack(x, slim.fully_connected, [4096, 1024], scope='fc')
            out = slim.fully_connected(x, 1, activation_fn=None, scope='fc_out')
        return out

    def classify(self, affinity_pred, name, is_training=True, reuse=False):
        """ Predicts whether the 2 image patches are from the same image """
        with tf.variable_scope(name, reuse=reuse):
            x = slim.stack(affinity_pred, slim.fully_connected, [512], scope='fc')
            out = slim.fully_connected(x, 1, activation_fn=None, scope='fc_out')
        return out

def initialize(args):
    return EXIFNet(**args)
