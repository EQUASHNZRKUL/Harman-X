import numpy as np 
import math
import tensorflow as tf 
import layers as l
import tf.nn
import os

class VGG:
  def __init__(self, dir):
    self.datadict = {}
    for filename in os.listdir(dir):
      if filename.find(".DS_Store") != -1: 
        dic = np.load(filename)
        v = dic["arr_0"]
        key = filename[:-4]
        self.datadict[key] = []
        for elt in v:
          self.datadict[key].append(elt)

  def build(self, input):

  def load_file(self, npz):
    key = npz[:-4]
    try:
      val = self.datadict[key]
    except KeyError:
      val = []
    dic = np.load(npz)
    v = dic["arr_0"]
    for elt in v:
      self.datadict[key] = val.append(elt)