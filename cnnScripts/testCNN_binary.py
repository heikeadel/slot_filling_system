#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
import os
import time
import numpy
from utils import readConfig, readWordvectors, getInput
from math import log
import cPickle
import string

import theano
import theano.tensor as T
from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

class CNN:

  def __init__(self, configfile, train = False):

    self.config = readConfig(configfile)

    self.addInputSize = 1
    logger.info("additional mlp input")

    wordvectorfile = self.config["wordvectors"]
    logger.info("wordvectorfile " + wordvectorfile)
    networkfile = self.config["net"]
    logger.info("networkfile " + networkfile)
    hiddenunits = int(self.config["hidden"])
    logger.info("hidden units " + str(hiddenunits))
    learning_rate = float(self.config["lrate"])
    logger.info("learning rate " + str(learning_rate))
    if train:
      self.batch_size = int(self.config["batchsize"])
    else:
      self.batch_size = 1
    logger.info("batch size " + str(self.batch_size))
    self.filtersize = [1,int(self.config["filtersize"])]
    nkerns = [int(self.config["nkerns"])]
    logger.info("nkerns " + str(nkerns))
    pool = [1, int(self.config["kmax"])]

    self.contextsize = int(self.config["contextsize"])
    logger.info("contextsize " + str(self.contextsize))

    if self.contextsize < self.filtersize[1]:
      logger.info("setting filtersize to " + str(self.contextsize))
      self.filtersize[1] = self.contextsize
    logger.info("filtersize " + str(self.filtersize))

    sizeAfterConv = self.contextsize - self.filtersize[1] + 1

    sizeAfterPooling = -1
    if sizeAfterConv < pool[1]:
      logger.info("setting poolsize to " + str(sizeAfterConv))
      pool[1] = sizeAfterConv
    sizeAfterPooling = pool[1]
    logger.info("kmax pooling: k = " + str(pool[1]))

    # reading word vectors
    self.wordvectors, self.vectorsize = readWordvectors(wordvectorfile)

    self.representationsize = self.vectorsize + 1

    rng = numpy.random.RandomState(23455)
    if train:
      seed = rng.get_state()[1][0]
      logger.info("seed: " + str(seed))

    # allocate symbolic variables for the data
    self.index = T.lscalar()  # index to a [mini]batch
    self.xa = T.matrix('xa')   # left context
    self.xb = T.matrix('xb')   # middle context
    self.xc = T.matrix('xc')   # right context
    self.y = T.ivector('y')    # label (only present in training)
    ishape = [self.representationsize, self.contextsize]  # this is the size of context matrizes

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    logger.info('... building the model')

    # Reshape input matrix to be compatible with our LeNetConvPoolLayer
    layer0a_input = self.xa.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0b_input = self.xb.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0c_input = self.xc.reshape((self.batch_size, 1, ishape[0], ishape[1]))

    # Construct convolutional pooling layer:
    filter_shape = (nkerns[0], 1, self.representationsize, self.filtersize[1])
    poolsize=(pool[0], pool[1])

    fan_in = numpy.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
              numpy.prod(poolsize))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    # the convolution weight matrix
    convW = theano.shared(numpy.asarray(
           rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
           dtype=theano.config.floatX),
                               borrow=True)

    # the bias is a 1D tensor -- one bias per output feature map
    b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
    convB = theano.shared(value=b_values, borrow=True)

    self.layer0a = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0a_input,
            image_shape=(self.batch_size, 1, ishape[0], ishape[1]),
            filter_shape=filter_shape, poolsize=poolsize)
    self.layer0b = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0b_input,
            image_shape=(self.batch_size, 1, ishape[0], ishape[1]),
            filter_shape=filter_shape, poolsize=poolsize)
    self.layer0c = LeNetConvPoolLayer(rng, W=convW, b=convB, input=layer0c_input,
            image_shape=(self.batch_size, 1, ishape[0], ishape[1]),
            filter_shape=filter_shape, poolsize=poolsize)

    layer0_output = T.concatenate([self.layer0a.output, self.layer0b.output, self.layer0c.output], axis = 3)

    layer2_input = layer0_output.flatten(2)

    self.additionalFeatures = T.matrix('additionalFeatures')
    additionalFeatsShaped = self.additionalFeatures.reshape((self.batch_size, 1))
    layer2_input = T.concatenate([layer2_input, additionalFeatsShaped], axis = 1)

    self.layer2 = HiddenLayer(rng, input=layer2_input, n_in=3 * nkerns[0] * sizeAfterPooling + self.addInputSize, n_out=hiddenunits, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    self.layer3 = LogisticRegression(input=self.layer2.output, n_in=hiddenunits, n_out=2)

    # create a list of all model parameters
    self.paramList = [self.layer0a.params, self.layer2.params, self.layer3.params]
    self.params = []
    for p in self.paramList:
      self.params += p
      logger.info(p) 

    if not train:
      self.gotNetwork = 1
      # load parameters
      if not os.path.isfile(networkfile):
        logger.error("network file does not exist")
        self.gotNetwork = 0
      else:
        netfile = open(networkfile)
        for p in self.params:
          p.set_value(cPickle.load(netfile), borrow=True)
        netfile.close()

  def classify(self, candidateAndFillerAndOffsetList):
    ##############
    # TEST MODEL #
    ##############

    logger.info('... testing')

    index = T.lscalar()  # index to a [mini]batch

    if self.gotNetwork == 0:
      return []

    inputMatrixDev_a, inputMatrixDev_b, inputMatrixDev_c, lengthListDev_a, lengthListDev_b, lengthListDev_c, inputFeaturesDev, _ = getInput(candidateAndFillerAndOffsetList, self.representationsize, self.contextsize, self.filtersize, self.wordvectors, self.vectorsize)
    # create input matrix and save them in valid_set

    dt = theano.config.floatX
    valid_set_xa = theano.shared(numpy.matrix(inputMatrixDev_a, dtype = dt))
    valid_set_xb = theano.shared(numpy.matrix(inputMatrixDev_b, dtype = dt))
    valid_set_xc = theano.shared(numpy.matrix(inputMatrixDev_c, dtype = dt))
    valid_mlp = theano.shared(numpy.matrix(inputFeaturesDev, dtype = dt))

    # compute number of minibatches for testing
    n_valid_batches = valid_set_xa.get_value(borrow=True).shape[0]
    n_valid_batches /= self.batch_size

    test_model_confidence = theano.function([index], self.layer3.results(),
            givens={
               self.xa: valid_set_xa[index * self.batch_size: (index + 1) * self.batch_size],
               self.xb: valid_set_xb[index * self.batch_size: (index + 1) * self.batch_size],
               self.xc: valid_set_xc[index * self.batch_size: (index + 1) * self.batch_size],
               self.additionalFeatures: valid_mlp[index * self.batch_size: (index + 1) * self.batch_size]})

    resultList = [test_model_confidence(i) for i in xrange(n_valid_batches)]

    return resultList
