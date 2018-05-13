## Fighting Fake News: Image Splice Detection via Learned Self-Consistency
### [[paper]](https://arxiv.org/pdf/1805.04096.pdf) [[website]](https://minyoungg.github.io/selfconsistency/)

[Minyoung Huh *<sup>1, 2</sup>](https://minyounghuh.com), [Andrew Liu *<sup>1</sup>](http://andrewhliu.github.io/), [Andrew Owens<sup>1</sup>](http://andrewowens.com/), [Alexei A. Efros<sup>1</sup>](https://people.eecs.berkeley.edu/~efros/)  
UC Berkeley, Berkeley AI Research<sup>1</sup>  
Carnegie Mellon University<sup>2</sup> 
### Abstract
In this paper, we introduce a self-supervised method for
learning to detect visual manipulations using only unlabeled data. Given a large collection of real photographs with automatically recorded EXIF meta-data, we train a model to determine whether an image is self-consistent -- that is, whether its content could have been produced by a single imaging pipeline.
    
### 1) Prerequisites
First clone this repo  
```git clone --single-branch https://github.com/minyoungg/selfconsistency```

All prerequisites should be listed in requirements.txt. The code is written on TensorFlow and is run on Python2.7, we have not verified whether Python3 works. The following command should automatically load any necessary requirements:  
```bash pip install -r requirements.txt```

### 2) Downloading pretrained model
To download our pretrained-model run the following script in the terminal:   
```chmod 755 download_model.sh && ./download_model.sh ```

### 3) Demo
To run our model on an image run the following code:   
``` python demo.py --im_path=./images/demo.png```

We have setup a ipython notebook demo [here](demo.ipynb)   
Disclaimer: Our model works the best on high-resolution natural images. Frames from a videos do not generally work well.

### Citation
If you find our work useful, please cite:   
```
@article{huh18forensics,
    title = {Fighting Fake News: Image Splice Detection via Learned Self-Consistency}
    author = {Huh, Minyoung and Liu, Andrew and
              Owens, Andrew and Efros, Alexei A.},
    journal = {arXiv preprint arXiv:1805.04096},
    year = {2018}
}
```

## Questions  
For any further questions please contact Minyoung Huh or Andrew Liu
