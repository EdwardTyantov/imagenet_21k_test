# imagenet_21k_test
A script for testing imagenet-21k inception model (https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-21k-inception.md)

#Install
- install mxnet: http://mxnet.readthedocs.io/en/latest/how_to/build.html (don't forget to compile with cuda flags)
- install python2.7
- install skimage for python

```
pip install scikit-image
```

- download and untar model

```
wget http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz; tar -xf inception-21k.tar.gz
```

#Test

launch:
```
python eval.py images/trump.jpg
```

result:
```
('Original Image Shape: ', (371, 660, 3))
('Top1: ', 'n09618880 Ex-president')
('Top5: ', ['n09618880 Ex-president', 'n10467179 President', 'n09917481 Chief Secretary', 'n10467395 President of the United States, United States President, President, Chief Executive', 'n10320863 Minister, government minister'])
```
