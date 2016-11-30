#!/usr/bin/env ipython 

from __future__ import print_function
import os, sys, urllib

if len(sys.argv) < 2:
  print("Usage:", sys.argv[0], "<url>")
  exit(0)

url = sys.argv[1]

import mxnet as mx

def download(url,prefix=''):
  filename = prefix+url.split("/")[-1]
  if not os.path.exists(filename):
    urllib.urlretrieve(url, filename)

path='http://data.mxnet.io/models/imagenet-11k/'
download(path+'resnet-152/resnet-152-symbol.json', 'full-')
download(path+'resnet-152/resnet-152-0000.params', 'full-')
download(path+'synset.txt', 'full-')

with open('full-synset.txt', 'r') as f:
  synsets = [l.rstrip() for l in f]

sym, arg_params, aux_params = mx.model.load_checkpoint('full-resnet-152', 0)

mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)

get_ipython().magic(u'matplotlib auto')
import matplotlib
matplotlib.rc("savefig", dpi=100)
import cv2
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=True):
  filename = url.split("/")[-1]
  urllib.urlretrieve(url, filename)
  img = cv2.imread(filename)
  if img is None:
    print('failed to download ' + url)
  return filename

def predict(filename, mod, synsets):
  img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
  if img is None:
    return None
  img = cv2.resize(img, (224, 224))
  img = np.swapaxes(img, 0, 2)
  img = np.swapaxes(img, 1, 2) 
  img = img[np.newaxis, :] 
    
  mod.forward(Batch([mx.nd.array(img)]))
  prob = mod.get_outputs()[0].asnumpy()
  prob = np.squeeze(prob)

  a = np.argsort(prob)[::-1]    
  for i in a[0:5]:
    print('probability=%f, class=%s' %(prob[i], synsets[i]))

results = predict(get_image(url), mod, synsets)
print(url)
print(results)
