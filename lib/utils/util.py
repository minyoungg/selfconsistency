# Shared and common functions (declustering redundant code)
import numpy as np, os
import random, cv2
import operator 

def get(link, save_as=False):
    import urllib
    base_dir = './tmp'
    assert type(link) == str, type(link)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    if save_as:
        save_path = os.path.join(base_dir, save_as)
    else:
        save_path = os.path.join(base_dir, 'tmp.png')
        
    urllib.urlretrieve(link, save_path)
    im = cv2.imread(save_path)[:,:,[2,1,0]]
    return im

def softmax(X, theta = 1.0, axis = None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p

def sort_dict(d, sort_by='value'):
    """ Sorts dictionary """
    assert sort_by in ['value', 'key'], sort_by
    if sort_by == 'key':
        return sorted(d.items(), key=operator.itemgetter(0))
    if sort_by == 'value':
        return sorted(d.items(), key=operator.itemgetter(1))

def random_crop(im, crop_size, return_crop_loc=False):
    """ Randomly crop """
    h,w = np.shape(im)[:2]
    hSt = random.randint(0, h - crop_size[0])
    wSt = random.randint(0, w - crop_size[1])
    patch = im[hSt:hSt+crop_size[0], wSt:wSt+crop_size[1], :]
    assert tuple(np.shape(patch)[:2]) == tuple(crop_size)
    if return_crop_loc:
        return patch, (hSt, wSt)
    return patch

def process_im(im):
    """ Normalizes images into the range [-1.0, 1.0] """
    im = np.array(im)
    if np.max(im) <= 1:
        # PNG format
        im = (2.0 * im) - 1.0
    else:
        # JPEG format
        im =  2.0 * (im / 255.) - 1.0
    return im

def deprocess_im(im, dtype=None):
    """ Map images in [-1.0, 1.0] back to [0, 255] """
    im = np.array(im)
    return ((255.0 * (im + 1.0))/2.0).astype(dtype)

def random_resize(im_a, im_b, same):
    valid_interps = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]

    def get_param():
        hr, wr = np.random.choice(np.linspace(0.5, 1.5, 11), 2)
        #hr, wr = np.random.uniform(low=0.5, high=1.5, size=2)
        interp = np.random.choice(valid_interps)
        return [hr, wr, interp]

    if same:
        if np.random.randint(2):
            a_par = get_param()
            im_a = cv2.resize(im_a, None, fx=a_par[0], fy=a_par[1], interpolation=a_par[2])
            im_b = cv2.resize(im_b, None, fx=a_par[0], fy=a_par[1], interpolation=a_par[2])
    else:
        a_par = get_param()
        im_a = cv2.resize(im_a, None, fx=a_par[0], fy=a_par[1], interpolation=a_par[2])
        if np.random.randint(2):
            b_par = get_param()
            while np.all(a_par == b_par):
                b_par = get_param()
            im_b = cv2.resize(im_b, None, fx=b_par[0], fy=b_par[1], interpolation=b_par[2])
    return im_a, im_b

def random_jpeg(im_a, im_b, same):
    def get_param():
        #jpeg_quality_a = np.random.randint(50, 100) # doesnt include 100
        return np.random.choice(np.linspace(50, 100, 11))

    if same:
        if np.random.randint(2):
            a_par = get_param()
            _, enc_a = cv2.imencode('.jpg', im_a, [int(cv2.IMWRITE_JPEG_QUALITY), a_par])
            im_a = cv2.imdecode(enc_a, 1)
            _, enc_b = cv2.imencode('.jpg', im_b, [int(cv2.IMWRITE_JPEG_QUALITY), a_par])
            im_b = cv2.imdecode(enc_b, 1)
    else:
        a_par = get_param()
        _, enc_a = cv2.imencode('.jpg', im_a, [int(cv2.IMWRITE_JPEG_QUALITY), a_par])
        im_a = cv2.imdecode(enc_a, 1)
        if np.random.randint(2):
            b_par = get_param()
            while np.all(a_par == b_par):
                b_par = get_param()
            _, enc_b = cv2.imencode('.jpg', im_b, [int(cv2.IMWRITE_JPEG_QUALITY), b_par])
            im_b = cv2.imdecode(enc_b, 1)
    return im_a, im_b

def gaussian_blur(im, kSz=None, sigma=1.0):
    # 5x5 kernel blur
    if kSz is None:
        kSz = np.ceil(3.0 * sigma)
        kSz = kSz + 1 if kSz % 2 == 0 else kSz
        kSz = max(kSz, 3) # minimum kernel size
    kSz = int(kSz)
    blur = cv2.GaussianBlur(im,(kSz,kSz), sigma)
    return blur

def random_blur(im_a, im_b, same):
    # only square gaussian kernels
    def get_param():
        kSz = (2 * np.random.randint(1, 8)) + 1 # [3, 15]
        sigma = np.random.choice(np.linspace(1.0, 5.0, 9))
        #sigma = np.random.uniform(low=1.0, high=5.0, size=None) # 3 * sigma = kSz
        return [kSz, sigma]

    if same:
        if np.random.randint(2):
            a_par = get_param()
            im_a = cv2.GaussianBlur(im_a, (a_par[0], a_par[0]), a_par[1])
            im_b = cv2.GaussianBlur(im_b, (a_par[0], a_par[0]), a_par[1])
    else:
        a_par = get_param()
        im_a = cv2.GaussianBlur(im_a, (a_par[0], a_par[0]), a_par[1])
        if np.random.randint(2):
            b_par = get_param()
            while np.all(a_par == b_par):
                b_par = get_param()
            im_b = cv2.GaussianBlur(im_b, (b_par[0], b_par[0]), b_par[1])
    return im_a, im_b

def random_noise(im):
    noise = np.random.randn(*np.shape(im)) * 10.0
    return np.array(np.clip(noise + im, 0, 255.0), dtype=np.uint8)
