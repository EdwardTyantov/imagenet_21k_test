# imagenet_21k_test
A script for testing imagenet-21k inception model (https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md)

#Install
1. install mxnet: http://mxnet.readthedocs.io/en/latest/how_to/build.html
2. install python2.7
3. install skimage for python: pip install scikit-image
4. download and unpack model: wget http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz; tar -xf inception-21k.tar.gz
5. test an image by launching: python eval.py images/trump.jpg
