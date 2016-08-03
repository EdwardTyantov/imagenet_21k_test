# imagenet_21k_test
A script for testing imagenet-21k inception model (https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md)

#Install
- install mxnet: http://mxnet.readthedocs.io/en/latest/how_to/build.html
- install python2.7
- install skimage for python

```
pip install scikit-image
```

- download and untar model

```
wget http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz; tar -xf inception-21k.tar.gz
```

- test an image by launching

```
python eval.py images/trump.jpg
```
