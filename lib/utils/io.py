import os
import json

def make_subdir(path):
    """ Makes subdirectories for the filepath """
    subdir = '/'.join(path.split('/')[:-1])
    if os.path.exists(subdir):
        return
    os.makedirs(subdir)
    return subdir

def make_dir(dir):
    """ Makes subdirectories for the filepath """
    if os.path.exists(dir):
        return dir
    os.makedirs(dir)
    return dir

def read_json(path):
    """
    loads all json line formatted file
    """
    data = [json.loads(line) for line in open(path)]
    return data

def to_npy(path):
    """ Changes to image path to npy format """
    return '.'.join(path.split('.')[:-1])+'.npy'

def show(args, phase, iter):
    """ Used to show training outputs """
    print '(%s) Iterations %i' % (phase, iter)
    max_len = max([len(k[0]) for k in args])
    for out in args:
        a,b = out
        print '\t',a.ljust(max_len),': ', b
    return

def add_summary(writer, list_of_summary, i):
    """ Adds list of summary to the writer """
    for s in list_of_summary:
        writer.add_summary(s, i)

def parse_checkpoint(ckpt):
    """ Parses checkpoint string to get iteration """
    assert type(ckpt) == str, ckpt
    try:
        i = int(ckpt.split('_')[-1].split('.')[0])
    except:
        print 'unknown checkpoint string format %s setting iteration to 0' % ckpt
        i = 0
    return i

def make_ckpt(saver, sess, save_prefix, i=None):
    """ Makes a checkpoint """
    if i is not None:
        save_prefix += '_' + str(i)
    save_path = save_prefix + '.ckpt'
    saver.save(sess, save_path)
    print 'Saved checkpoint at %s' % save_path
    return
