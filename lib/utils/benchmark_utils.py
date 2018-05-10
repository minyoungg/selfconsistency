import sys, os
import init_paths
from lib.utils import queue_runner, util
import tensorflow as tf
import threading
import numpy as np
import time
import scipy.misc
import cv2


class EfficientBenchmark():
    
    def __init__(self, solver, net_module_obj, net_module_obj_init_params, im,
                 num_processes=1, num_threads=1, stride=None, max_bs=20000, n_anchors=3,
                 patch_size=224, auto_close_sess=True, patches=None, mirror_pred=False,
                 dense_compute=False, num_per_dim=30):
        """
        solver: The model solver to run predictions
        net_module_obj: The corresponding net class
        net_module_obj_init_params: Dictionary that would normally be passed into net.initialize
        im: The image to analyze
        num_processes: Number of data grabbing processes, can only run 1
        num_thread: Number of threads to trasnfer from Python Queue to TF Queue
        stride: Distance between sampled grid patches
        max_bs: For precomputing, determines number of index selecting we do per batch
        n_anchors: Number of anchor patches, if 
        patches: A numpy array of (n x patch_size x patch_size x 3) which is used as anchor patch,
            should n_anchors argument
        auto_close_sess: Whether to close tf session after finishing analysis
        
        (deprecated):
        dense_compute, always leave on false, precomputing does dense faster
        mirror_pred, always leave on false, precomputing does mirror predictions
        """
        assert num_processes == 1, "Can only do single process"
        assert num_threads > 0, "Need at least one threads for queuing"
        
        self.use_patches = False
        if type(patches) != type(None):
            # use defined patches
            assert patches.shape[0] == n_anchors
            self.use_patches = True
            self.patches = patches
            
        self.mirror_pred = mirror_pred
        # For when we use indices for precomputed features
        self.max_bs = max_bs
        
        self.solver = solver
        self.n_anchors = n_anchors
        self.num_per_dim = num_per_dim
        self.patch_size = patch_size
        self.recompute_stride = False
        self.stride = stride
        if not stride:
            # compute stride dynamically
            self.recompute_stride = True
            self.stride = self.compute_stride(im)
        self.dense_compute = dense_compute
        if dense_compute:
            self.patches = None
        self.num_processes = num_processes
        self.num_threads = num_threads
        
        self.label_shape = 1 if not 'num_classes' in net_module_obj_init_params else net_module_obj_init_params['num_classes']
        
        self.cr = self.update_queue_runner(im)
        self.auto_close_sess = auto_close_sess
        self.n_responses = self.max_h_ind * self.max_w_ind if self.dense_compute else n_anchors 
        
        net_module_obj_init_params["train_runner"] = self.cr
        net_module_obj_init_params["use_tf_threading"] = True
        self.net = net_module_obj.initialize(net_module_obj_init_params)
        self.solver.setup_net(net=self.net)
    
    def compute_stride(self, im):
        return (max(im.shape[0], im.shape[1]) - self.patch_size) // self.num_per_dim
    
    def update_queue_runner(self, im):
        # returns a new queue_runner
        self.set_image(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32))
        
        fn = self.dense_argless if self.dense_compute else self.argless 
        cr = queue_runner.CustomRunner(fn, n_processes=self.num_processes,
                                       n_threads=self.num_threads)
        self.original_cr_get_inputs = cr.get_inputs
        self.set_image(im)
        
        def new_cr(batch_size):
            self.anch_indices_, self.h_indices_, self.w_indices_, im_a, im_b = self.original_cr_get_inputs(batch_size)
            # we don't use the label since it's test time
            if self.label_shape == 1:
                return im_a, im_b, tf.zeros((batch_size), dtype=tf.int64)
            else:
                return im_a, im_b, tf.zeros((batch_size, self.label_shape), dtype=tf.float32)
            
        # Directly rewriting get_inputs since that's all the net class sees and uses
        cr.get_inputs = new_cr
        
        return cr
        
        
    def reset_image(self, im):
        if self.recompute_stride:
            # compute stride dynamically
            self.stride = self.compute_stride(im)
            
        fn = self.dense_argless if self.dense_compute else self.argless 
        self.cr.kill_programs()
        # programs are all dead, purge tf queue
        
        while True:
            self.solver.sess.run(self.cr.tf_queue.dequeue_up_to(self.cr.tf_queue.size()))
            remain = self.solver.sess.run(self.cr.tf_queue.size())
            if remain == 0:
                break
        
        self.set_image(im)
        self.cr.set_data_fn(fn)
        self.cr.start_p_threads(self.solver.sess)
    
    def get_patch(self, hind, wind):
        return self.image[hind:hind+self.patch_size, wind:wind+self.patch_size]
        
    def rand_patch(self):
        h = np.random.randint(self.image.shape[0] - self.patch_size + 1)
        w = np.random.randint(self.image.shape[1] - self.patch_size + 1)
        return self.image[h:h+self.patch_size, w:w+self.patch_size, :]
        
    def get_anchor_patches(self):
        # set seed here if want same patches
        # Regardless of whether use or not, should
        # be 0 if not dense compute
        self.anchor_count = 0
        if self.dense_compute:
            self.anchor_inds = self.indices.copy()
            return util.process_im(np.array([self.get_patch(
                self.anchor_inds[i][0], self.anchor_inds[i][1]) for i in range(self.anchor_count,
                                                                     min(self.anchor_count + self.n_anchors,
                                                                         self.anchor_inds.shape[0]))]))
        
        if self.use_patches:
            # pass existing patches
            return util.process_im(self.patches)
        return util.process_im(
            np.array([self.rand_patch() for i in range(self.n_anchors)], dtype=np.float32))
    
    def set_image(self, image):
        # new image, need to refresh
        self.image = image
        self.max_h_ind = 1 + int(np.floor((self.image.shape[0] - self.patch_size) / float(self.stride)))
        self.max_w_ind = 1 + int(np.floor((self.image.shape[1] - self.patch_size) / float(self.stride)))
        self.indices = np.mgrid[0:self.max_h_ind, 0:self.max_w_ind].reshape((2, -1)).T # (n 2)
        self.anchor_patches = self.get_anchor_patches()
        self.count = -1
    
    def data_fn(self, hind, wind):
        n_anchors = self.anchor_patches.shape[0]
        y_ind, x_ind = hind * self.stride, wind * self.stride
        
        patch = self.image[y_ind:y_ind + self.patch_size, 
                           x_ind:x_ind + self.patch_size,
                           :]
            
        anchor_inds = np.arange(self.anchor_count, self.anchor_count + n_anchors)
        h_inds = np.array([hind] * n_anchors, dtype=np.int64)
        w_inds = np.array([wind] * n_anchors, dtype=np.int64)
        batch_a = self.anchor_patches
        batch_b = util.process_im(np.array([patch] * n_anchors, dtype=np.float32))
        # anc, y, x, bat_a, bat_b
        if self.mirror_pred:
            anchor_inds = np.vstack([anchor_inds] * 2)
            h_inds = np.vstack([h_inds] * 2)
            w_inds = np.vstack([w_inds] * 2)
            batch_a, batch_b = np.vstack([batch_a, batch_b]), np.vstack([batch_b, batch_a])
            
        return anchor_inds, h_inds, w_inds, batch_a, batch_b
    
    def dense_argless(self):
        assert False, "Deprecated"
        if self.count >= self.indices.shape[0]:
            self.count = 0
            self.anchor_count += self.n_anchors
            if self.anchor_count >= self.anchor_inds.shape[0]:
                raise StopIteration()
            inds2 = self.anchor_inds[self.anchor_count]
            self.anchor_patches =  util.process_im(np.array([self.get_patch(
                self.anchor_inds[i][0], self.anchor_inds[i][1]) for i in range(self.anchor_count,
                                                                     min(self.anchor_count + self.n_anchors,
                                                                         self.anchor_inds.shape[0]))]))
            self.n_anchors = self.anchor_patches.shape[0]
        inds = self.indices[self.count]
        self.count += 1
        d = self.data_fn(inds[0], inds[1])
        return d
        
    
    def argless(self):
        self.count += 1
        if self.count >= self.indices.shape[0]:
            raise StopIteration()
        inds = self.indices[self.count]
        return self.data_fn(inds[0], inds[1])
    
    def argless_extract_inds(self):
        iterator = np.mgrid[0:self.max_h_ind, 0:self.max_w_ind, 0:self.max_h_ind, 0:self.max_w_ind].reshape((4, -1)).T # (n 4)
        count = 0
        while True:
            if count * self.max_bs > len(iterator):
                break
            # each indice is a read into np.mgrid[0:self.max_h_ind, 0:self.max_w_ind]
            yield iterator[count * self.max_bs:(count + 1) * self.max_bs, :] # self.max_bs x 4
            count += 1

    def run_ft(self, num_fts=4096):
        #print("Starting Analysis")
        # Batch_b contains the sweeping patches, feat_b to get features of a patch
        # For most efficient running set n_anchors to 1
        responses = np.zeros((num_fts, self.max_h_ind, 
                              self.max_w_ind))  
        
        expected_num_running = self.max_h_ind * self.max_w_ind
        visited = np.zeros((self.max_h_ind, self.max_w_ind))
        while True:
            try:
                # t0 = time.time()
                h_ind_, w_ind_, fts_ = self.solver.sess.run([self.h_indices_,
                                                             self.w_indices_,
                                                             self.solver.net.im_b_feat])
                # print time.time() - t0
                for i in range(h_ind_.shape[0]):
                    responses[:, h_ind_[i], w_ind_[i]] = fts_[i]
                    visited[h_ind_[i], w_ind_[i]] = 1
                if np.sum(visited) == expected_num_running:
                    raise RuntimeError("Finished")
                    
            except tf.errors.OutOfRangeError as e:
                # TF Queue emptied, return responses
                if self.auto_close_sess:
                    self.solver.sess.close()
                return responses
            except RuntimeError as e:
                if self.auto_close_sess:
                    self.solver.sess.close()
                return responses
        
    def precomputed_analysis_vote_cls(self, num_fts=4096):
        #print("Starting Analysis")
        assert not self.auto_close_sess, "Need to keep sess open"
        
        feature_response = self.run_ft(num_fts=num_fts)
        
        flattened_features = feature_response.reshape((num_fts, -1)).T
        # Use np.unravel_index to recover x,y coordinate
        
        spread = max(1, self.patch_size // self.stride)
        
        responses = np.zeros((self.max_h_ind + spread - 1, self.max_w_ind + spread - 1,
                              self.max_h_ind + spread - 1, self.max_w_ind + spread - 1), dtype=np.float32)
        vote_counts = np.zeros((self.max_h_ind + spread - 1, self.max_w_ind + spread - 1,
                                self.max_h_ind + spread - 1, self.max_w_ind + spread - 1)) + 1e-4
        
        iterator = self.argless_extract_inds()
        while True:
            try:
                inds = next(iterator)
            except StopIteration as e:
                if self.auto_close_sess:
                    self.solver.sess.close()
                out = (responses / vote_counts)
                return out
            patch_a_inds = inds[:, :2]
            patch_b_inds = inds[:, 2:]

            a_ind = np.ravel_multi_index(patch_a_inds.T, [self.max_h_ind, self.max_w_ind])
            b_ind = np.ravel_multi_index(patch_b_inds.T, [self.max_h_ind, self.max_w_ind])

            # t0 = time.time()
            preds_ = self.solver.sess.run(self.solver.net.pc_cls_pred,
                                          feed_dict={self.net.precomputed_features:flattened_features,
                                                     self.net.im_a_index: a_ind,
                                                     self.net.im_b_index: b_ind})
            # print preds_
            # print time.time() - t0
            for i in range(preds_.shape[0]):
                responses[inds[i][0] : (inds[i][0] + spread),
                          inds[i][1] : (inds[i][1] + spread),
                          inds[i][2] : (inds[i][2] + spread),
                          inds[i][3] : (inds[i][3] + spread)] += preds_[i]
                vote_counts[inds[i][0] : (inds[i][0] + spread),
                          inds[i][1] : (inds[i][1] + spread),
                          inds[i][2] : (inds[i][2] + spread),
                          inds[i][3] : (inds[i][3] + spread)] += 1
                

