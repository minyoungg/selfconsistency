from __future__ import print_function
from __future__ import division

import os, sys, numpy as np, ast
import init_paths
import load_models
from lib.utils import benchmark_utils, util
import tensorflow as tf
import cv2, time, scipy, scipy.misc as scm, sklearn.cluster, skimage.io as skio, numpy as np, argparse
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def mean_shift(points_, heat_map, iters=5):
    points = np.copy(points_)
    kdt = scipy.spatial.cKDTree(points)
    eps_5 = np.percentile(scipy.spatial.distance.cdist(points, points, metric='euclidean'), 10)
    
    for epis in range(iters):
        for point_ind in range(points.shape[0]):
            point = points[point_ind]
            nearest_inds = kdt.query_ball_point(point, r=eps_5)
            points[point_ind] = np.mean(points[nearest_inds], axis=0)
    val = []
    for i in range(points.shape[0]):
        val.append(kdt.count_neighbors(scipy.spatial.cKDTree(np.array([points[i]])), r=eps_5))
    mode_ind = np.argmax(val)
    ind = np.nonzero(val == np.max(val))
    return np.mean(points[ind[0]], axis=0).reshape(heat_map.shape[0], heat_map.shape[1])

def centroid_mode(heat_map):
    eps_thresh = np.percentile(heat_map, 10)
    k = heat_map <= eps_thresh
    # Get's max centroid
    num_affinities = np.sum(k, axis=(2, 3))
    x = np.nonzero(num_affinities >= np.max(num_affinities))
    if type(x) is tuple:
        ind1 = x[0][0]
        ind2 = x[1][0]
    else:
        ind1 = x[0]
        ind2 = x[1]
    assert np.max(num_affinities) == num_affinities[ind1, ind2]
    return heat_map[ind1, ind2]

def normalized_cut(res):
    sc = sklearn.cluster.SpectralClustering(n_clusters=2, n_jobs=-1,
                                            affinity="precomputed")
    out = sc.fit_predict(res.reshape((res.shape[0] * res.shape[1], -1)))
    vis = out.reshape((res.shape[0], res.shape[1]))
    return vis
def process_response_no_resize(response):
    return 255 * plt.cm.jet(response)[:,:,:3]

def process_response(response):
    size = get_resized_shape(response)
    im = 255 * plt.cm.jet(response)[:,:,:3]
    return scm.imresize(im, size)# , interp='nearest')

def get_resized_shape(im, max_im_dim=400):
    ratio = float(max_im_dim) / np.max(im.shape)
    return (int(im.shape[0] * ratio), int(im.shape[1] * ratio), 3)

def process_image(im):
    size = get_resized_shape(im)
    return scm.imresize(im, size) #, interp='nearest')

def norm(response):
    res = response - np.min(response)
    return res/np.max(res)

def apply_mask(im, mask):
    mask = scipy.misc.imresize(mask, (im.shape[0], im.shape[1])) / 255.
    mask = mask.reshape(im.shape[0], im.shape[1], 1)
    mask = mask * 0.8 + 0.2
    return mask * im

def aff_fn(v1, v2):
    return np.mean((v1 * v2 + (1 - v1)*(1 - v2)))

def ssd_distance(results, with_inverse=True):
    def ssd(x, y):
        # uses mean instead
        return np.mean(np.square(x - y))
    
    results = np.array(results)
    results = np.concatenate([results, 1.0 - results], axis=0)
    
    dist_matrix = np.zeros((len(results), len(results)))
    for i, r_x in enumerate(results):
        for j, r_y in enumerate(results):
            score = ssd(r_x, r_y)
            dist_matrix[i][j] = score 
    return dist_matrix, results

def dbscan_consensus(results, eps_range=(0.1, 0.5), eps_sample=10, dbscan_sample=4):
    """
    Slowly increases DBSCAN epsilon until a cluster is found. 
    The distance between responses is the SSD.
    Best prediction is based on the spread within the cluster. 
    Here spread is the average per-pixel variance of the output.
    The cluster is then combined using the median of the cluster.
    When no cluster is found, returns the response
    that has smallest median score across other responses.
    """
    
    dist_matrix, results = ssd_distance(results, with_inverse=True)
    
    debug = False #True
    lowest_spread = 100.0
    best_pred = None

    for eps in np.linspace(eps_range[0], eps_range[1], eps_sample):
        db = DBSCAN(eps=eps, min_samples=dbscan_sample).fit(dist_matrix)
        labels = set(db.labels_)
        
        if debug: 
            print('DBSCAN with epsilon %.3f' % eps)
            print('Found %i labels' % len(labels))
            
        try:
            labels.remove(-1)
        except:
            pass
        
        if debug: 
            print('%i Unique cluster' % len(labels))
        labels = np.array(list(labels))

        if len(labels) < 2:
            if debug: 
                print('Not enough cluster found')
            continue 

        clusters = {l:np.argwhere(db.labels_ == l) for l in labels}
        cluster_spreads = {}
        cluster_preds = {}

        for lbl, cluster_indices in clusters.items():
            if debug: 
                print('Cluster %i with %i samples' % (lbl, len(cluster_indices)))
                
            cluster_indices = np.squeeze(cluster_indices)
            cluster_results = [results[i] for i in cluster_indices]

            #mean_result   = np.mean(cluster_results, axis=0)
            median_result = np.median(cluster_results, axis=0)

            # Average Per pixel deviation
            average_spread = np.mean(np.std(cluster_results, axis=0))
            cluster_spreads[lbl] = average_spread
            cluster_preds[lbl] = median_result
            #print average_spread
            if average_spread < lowest_spread:
                lowest_spread = average_spread
                best_pred = median_result

        best_lbl, avg_spread = util.sort_dict(cluster_spreads)[0]

        if debug: 
            print('Cluster spread %.3f' % avg_spread)
            plt.imshow(cluster_preds[best_lbl], cmap='jet', vmin=0.0, vmax=1.0)  
            plt.show()

    if best_pred is None:
        # Uses a sample that has the median minimum distance between all predicted sample
        print('Failed to find DBSCAN cluster')
        compact_dist_matrix = dist_matrix[:len(dist_matrix)//2, :len(dist_matrix)//2]
        avg_dist = np.median(compact_dist_matrix, axis=0)
        best_pred = results[np.argmin(avg_dist)]
    
    if debug:
        plt.figure()
        plt.imshow(best_pred, cmap='jet', vmin=0.0, vmax=1.0)  
    return best_pred, lowest_spread

def run_vote_no_threads(image, solver, exif_to_use, n_anchors=1, num_per_dim=None,
             patch_size=None, batch_size=None, sample_ratio=3.0, override_anchor=False):
    """
    solver: exif_solver module. Must be initialized and have a network connected.
    exif_to_use: exif to extract responses from. A list. If exif_to_use is None
                 extract result from classification output cls_pred
    n_anchors: number of anchors to use.
    num_per_dim: number of patches to use along the largest dimension.
    patch_size: size of the patch. If None, uses the one specified in solver.net
    batch_size: size of the batch. If None, uses the one specified in solver.net
    sample_ratio: The ratio of overlap between patches. num_per_dim must be None
                  to be useful.
    """

    h, w = np.shape(image)[:2]

    if patch_size is None:
        patch_size = solver.net.im_size

    if batch_size is None:
        batch_size = solver.net.batch_size

    if num_per_dim is None:
        num_per_dim = int(np.ceil(sample_ratio * (max(h,w)/float(patch_size))))

    if exif_to_use is None:
        not_exif = True
        exif_to_use = ['out']
    else:
        not_exif = False
        exif_map = {e: np.squeeze(np.argwhere(np.array(solver.net.train_runner.tags) == e)) for e in exif_to_use}

    responses   = {e:np.zeros((n_anchors, h, w)) for e in exif_to_use}
    vote_counts = {e:1e-6 * np.ones((n_anchors, h, w)) for e in exif_to_use}

    if np.min(image) < 0.0:
        # already preprocessed
        processed_image = image
    else:
        processed_image = util.process_im(image)
    ones = np.ones((patch_size, patch_size))

    anchor_indices = []
    # select n anchors
    for anchor_idx in range(n_anchors):
        if override_anchor is False:
            _h, _w  = np.random.randint(0, h - patch_size), np.random.randint(0, w - patch_size)
        else:
            assert len(override_anchor) == 2, override_anchor
            _h, _w = override_anchor

        anchor_indices.append((_h, _w))
        anchor_patch = processed_image[_h:_h+patch_size, _w:_w+patch_size, :]

        batch_a = np.tile([anchor_patch], [batch_size, 1, 1, 1])
        batch_b, batch_b_coord = [], []

        prev_batch = None
        for i in np.linspace(0, h - patch_size, num_per_dim).astype(int):
            for j in np.linspace(0, w - patch_size, num_per_dim).astype(int):
                compare_patch = processed_image[i:i+patch_size, j:j+patch_size]
                batch_b.append(compare_patch)
                batch_b_coord.append((i,j))

                if len(batch_b) == batch_size:
                    if not_exif:
                        pred = solver.sess.run(solver.net.cls_pred,
                                 feed_dict={solver.net.im_a:batch_a,
                                            solver.net.im_b:batch_b,
                                            solver.net.is_training:False})
                    else:
                        pred = solver.sess.run(solver.net.pred,
                                 feed_dict={solver.net.im_a:batch_a,
                                            solver.net.im_b:batch_b,
                                            solver.net.is_training:False})

                    for p_vec, (_i, _j) in zip(pred, batch_b_coord):
                        for e in exif_to_use:
                            if not_exif:
                                p = p_vec[0]
                            else:
                                p = p_vec[int(exif_map[e])]
                            responses[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += (p * ones)
                            vote_counts[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += ones
                    prev_batch = batch_b
                    batch_b, batch_b_coord = [], []

        if len(batch_b) > 0:
            batch_b_len = len(batch_b)
            to_pad = np.array(prev_batch)[:batch_size - batch_b_len]
            batch_b = np.concatenate([batch_b, to_pad], axis=0)

            if not_exif:
                pred = solver.sess.run(solver.net.cls_pred,
                                     feed_dict={solver.net.im_a:batch_a,
                                                solver.net.im_b:batch_b,
                                                solver.net.is_training:False})
            else:
                pred = solver.sess.run(solver.net.pred,
                                     feed_dict={solver.net.im_a:batch_a,
                                                solver.net.im_b:batch_b,
                                                solver.net.is_training:False})

            for p_vec, (_i, _j) in zip(pred, batch_b_coord):
                for e in exif_to_use:
                    if not_exif:
                        p = p_vec[0]
                    else:
                        p = p_vec[int(exif_map[e])]
                    responses[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += (p * ones)
                    vote_counts[e][anchor_idx, _i:_i+patch_size, _j:_j+patch_size] += ones

    return {e: {'responses':(responses[e] / vote_counts[e]), 'anchors':anchor_indices} for e in exif_to_use}

class Demo():
    def __init__(self, ckpt_path='/data/scratch/minyoungg/ckpt/exif_medifor/exif_medifor.ckpt', use_gpu=0,
                 quality=3.0, patch_size=128, num_per_dim=30):
        self.quality = quality # sample ratio
        self.solver, nc, params = load_models.initialize_exif(ckpt=ckpt_path, init=False, use_gpu=use_gpu)
        params["im_size"] = patch_size
        self.im_size = patch_size
        tf.reset_default_graph()
        im = np.zeros((256, 256, 3))
        self.bu = benchmark_utils.EfficientBenchmark(self.solver, nc, params, im, auto_close_sess=False, 
                                                     mirror_pred=False, dense_compute=False, stride=None, n_anchors=10,
                                                     patch_size=patch_size, num_per_dim=num_per_dim)
        return

    def run(self, im, gt=None, show=False, save=False,
            blue_high=False, use_ncuts=False):
        # run for every new image
        self.bu.reset_image(im)
        res = self.bu.precomputed_analysis_vote_cls(num_fts=4096)
        #print('result shape', np.shape(res))
        ms = mean_shift(res.reshape((-1, res.shape[0] * res.shape[1])), res)
        
        if np.mean(ms > .5) > .5:
            # majority of the image is above .5
            if blue_high:
                ms = 1 - ms
        
        if use_ncuts:

            ncuts = normalized_cut(res)
            if np.mean(ncuts > .5) > .5:
                # majority of the image is white
                # flip so spliced is white
                ncuts = 1 - ncuts
            out_ncuts = cv2.resize(ncuts.astype(np.float32), (im.shape[1], im.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        out_ms = cv2.resize(ms, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            
        if use_ncuts:
            return out_ms, out_ncuts
        return out_ms
    
    def run_vote(self, im, num_per_dim=3, patch_size=128):
        h,w = np.shape(im)[:2]
        all_results = []
        for hSt in np.linspace(0, h - patch_size, num_per_dim).astype(int):
            for wSt in np.linspace(0, w - patch_size, num_per_dim).astype(int):
                res = run_vote_no_threads(im, self.solver, None, n_anchors=1, num_per_dim=None,
                                          patch_size=128, batch_size=64, sample_ratio=self.quality, 
                                          override_anchor=(hSt, wSt))['out']['responses'][0]
                all_results.append(res)
                
        return dbscan_consensus(all_results)
    
    def __call__(self, url, dense=False):
        """
        @Args
            url: This can either be a web-url or directory
            dense: If False, runs the new DBSCAN clustering. 
                   Using dense will be low-res and low-variance.
        @Returns
            output of the clustered response
        """    
        if type(url) is not str:
            im = url
        else:
            if url.startswith('http'):
                im = util.get(url)
            else:
                im = cv2.imread(url)[:,:,[2,1,0]]

        #print('Image shape:', np.shape(im))
        assert min(np.shape(im)[:2]) > self.im_size, 'image dimension too small'
        
        if not dense:
            # Runs default dense clustering
            out, _ = self.run_vote(im, num_per_dim=3, patch_size=self.im_size)
        else:
            # Runs DBSCAN
            out = self.run(im)
        return im, out
    
if __name__ == '__main__':
    plt.switch_backend('agg')
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", type=str, help="path_to_image")
    cfg = parser.parse_args()
    
    assert os.path.exists(cfg.im_path)
    
    imid = cfg.im_path.split('/')[-1].split('.')[0]
    save_path = os.path.join('./images', imid + '_result.png')
    
    ckpt_path = './ckpt/exif_final/exif_final.ckpt'
    exif_demo = Demo(ckpt_path=ckpt_path, use_gpu=0, quality=3.0, num_per_dim=30)
    
    print('Running image %s' % cfg.im_path)
    ms_st = time.time()
    im_path = cfg.im_path
    im, res = exif_demo(im_path, dense=True)
    print('MeanShift run time: %.3f' % (time.time() - ms_st))
    
    plt.subplots(figsize=(16, 8))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(im)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Cluster w/ MeanShift')
    plt.axis('off')
    if np.mean(res > 0.5) > 0.5:
        res = 1.0 - res
    plt.imshow(res, cmap='jet', vmin=0.0, vmax=1.0)
    plt.savefig(save_path)
    print('Result saved %s' % save_path)
