import tensorflow as tf
import numpy as np
import time
import multiprocessing as mp
import threading
import Queue
            
class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
        
    # Need to call the following code block after initializing everything
    self.sess.run(tf.global_variables_initializer())

    if self.use_tf_threading:
        self.coord = tf.train.Coordinator()
        self.net.train_runner.start_p_threads(self.sess)
        tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
    """
    def __init__(self, arg_less_fn, override_dtypes=None,
                 n_threads=1, n_processes=3, max_size=30):
        # arg_less_fn should be function that returns already ready data
        # in the form of numpy arrays. The shape of the output is
        # used to shape the output tensors. Should be ready to call at init_time
        # override_dtypes is the typing, default to numpy's encoding.
        self.data_fn = arg_less_fn
        self.n_threads = n_threads
        self.n_processes = n_processes
        self.max_size = max_size
        self.use_pool = False
        
        # data_fn shouldn't take any argument,
        # just directly return the necessary data
        # set via the setter fn
        
        data = self.data_fn()
        self.inps = []
        shapes, dtypes = [], []
        for i, d in enumerate(data):
            inp = tf.placeholder(dtype=d.dtype, shape=[None] + list(d.shape[1:]))
            self.inps.append(inp)
            # remove batching index for individual element
            shapes.append(d.shape[1:])
            dtypes.append(d.dtype)
        # The actual queue of data.
        self.tf_queue = tf.FIFOQueue(shapes=shapes,
                                           # override_dtypes or default
                                           dtypes=override_dtypes or dtypes,
                                           capacity=2000)

        # The symbolic operation to add data to the queue
        self.enqueue_op = self.tf_queue.enqueue_many(self.inps)

    def get_inputs(self, batch_size):
        """
        Return's tensors containing a batch of images and labels
        
        if tf_queue has been closed this will raise a QueueBase exception
        killing the main process if a StopIteration is thrown in one of the
        data processes.
        """
        return self.tf_queue.dequeue_up_to(tf.reduce_min([batch_size, self.tf_queue.size()]))

    def thread_main(self, sess, stop_event):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        tt_last_update = time.time() - 501
        count = 0
        tot_p_end = 0
        processes_all_done = False
        while not stop_event.isSet():
            if tt_last_update + 500 < time.time():
                t = time.time()
                # 500 seconds since last update
                #print("DataQueue Threading Update:")
                #print("TIME: " + str(t))
                # MP.Queue says it is not thread safe and is not perfectly accurate.
                # Just want to make sure there's no leakage and max_size 
                # is safely hit
                #print("APPROX SIZE: %d" % self.queue.qsize())
                #print("TOTAL FETCH ITERATIONS: %d" % count)
                tt_last_update = t
            count += 1
            if processes_all_done and self.queue.empty():
                break
            try:
                data = self.queue.get(5)
            except Queue.Empty:
                continue
                
            if type(data) == type(StopIteration()):
                tot_p_end += 1
                if tot_p_end == self.n_processes:
                    # Kill any processes
                    # may need a lock here if multithreading
                    processes_all_done = True
                    #print("ALL PROCESSES DONE")
                continue
            
            fd = {}
            for i, d in enumerate(data):
                fd[self.inps[i]] = d
            sess.run(self.enqueue_op, feed_dict=fd)
        self.queue.close()

    def process_main(self, queue):
        # Scramble seed so it's not a copy of the parent's seed
        np.random.seed()
        # np.random.seed(1)
        try:
            while True:
                queue.put(self.data_fn())
        except StopIteration as e:
            # Should only manually throw when want to close queue
            queue.put(e)
            #raise e
            return
        
        except Exception as e:
            queue.put(StopIteration())
            #raise e
            return
        
        
    def set_data_fn(self, fn):
        self.data_fn = fn
        
    def start_p_threads(self, sess):
        """ Start background threads to feed queue """
        self.processes = []
        self.queue = mp.Queue(self.max_size)
        
        for n in range(self.n_processes):
            p = mp.Process(target=self.process_main, args=(self.queue,))
            p.daemon = True # thread will close when parent quits
            p.start()
            self.processes.append(p)
            
        self.threads = []
        self.thread_event_killer = []
        for n in range(self.n_threads):
            kill_thread = threading.Event()
            self.thread_event_killer.append(kill_thread)
            
            t = threading.Thread(target=self.thread_main, args=(sess, kill_thread))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)
        return self.processes + self.threads
    
    def kill_programs(self):
        # Release objects here if need to
        # threads should die in at least 5 seconds because
        # nothing blocks for more than 5 seconds
        
        # Sig term, kill first so no more data
        [p.terminate() for p in self.processes]
        [p.join() for p in self.processes]
        
        # kill second after purging
        [e.set() for e in self.thread_event_killer]
    