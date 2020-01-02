#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
import os
import numpy
from utils import readConfig, readWordvectors, getInput, getSlot2Types
import cPickle

import theano
import theano.tensor as T
from layers import LogisticRegression, HiddenLayer, LeNetConvPoolLayer

from sklearn.preprocessing import MultiLabelBinarizer

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
    logger.info("wordvectorfile " + str(wordvectorfile))
    networkfile = self.config["net"]
    logger.info("networkfile " + str(networkfile))
    hiddenunits = int(self.config["hidden"])
    logger.info("hidden units " + str(hiddenunits))
    hiddenunitsNER = 50
    if "hiddenunitsNER" in self.config:
      hiddenunitsNER = int(self.config["hiddenunitsNER"])
    logger.info("hidden units NER " + str(hiddenunitsNER))
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
    self.y = T.imatrix('y')   # label (only present in training)
    self.yNER1 = T.imatrix('yNER1') # label for first entity
    self.yNER2 = T.imatrix('yNER2') # label for second entity
    ishape = [self.representationsize, self.contextsize]  # this is the size of context matrizes

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    logger.info('... building the model')

    # Reshape input matrix to be compatible with our LeNetConvPoolLayer
    layer0a_input = self.xa.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0b_input = self.xb.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0c_input = self.xc.reshape((self.batch_size, 1, ishape[0], ishape[1]))

    self.y_reshaped = self.y.reshape((self.batch_size, 1))
    yNER1reshaped = self.yNER1.reshape((self.batch_size, 1))
    yNER2reshaped = self.yNER2.reshape((self.batch_size, 1))

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

    #layer0_output = T.concatenate([self.layer0a.output, self.layer0b.output, self.layer0c.output], axis = 3)
    layer0aflattened = self.layer0a.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0bflattened = self.layer0b.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0cflattened = self.layer0c.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0_output = T.concatenate([layer0aflattened, layer0bflattened, layer0cflattened], axis = 1)

    self.layer1a = HiddenLayer(rng = rng, input = self.yNER1, n_in = 6, n_out = hiddenunitsNER, activation = T.tanh)
    self.layer1b = HiddenLayer(rng = rng, input = self.yNER2, n_in = 6, n_out = hiddenunitsNER, activation = T.tanh, W = self.layer1a.W, b = self.layer1a.b)


    layer2_input = T.concatenate([layer0_output, self.layer1a.output, self.layer1b.output], axis = 1)
    layer2_inputSize = 3 * nkerns[0] * sizeAfterPooling + 2 * hiddenunitsNER

    self.additionalFeatures = T.matrix('additionalFeatures')
    additionalFeatsShaped = self.additionalFeatures.reshape((self.batch_size, 1))
    layer2_input = T.concatenate([layer2_input, additionalFeatsShaped], axis = 1)
    layer2_inputSize += self.addInputSize

    self.layer2 = HiddenLayer(rng, input=layer2_input, n_in=layer2_inputSize,
                         n_out=hiddenunits, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    self.layer3 = LogisticRegression(input=self.layer2.output, n_in=hiddenunits, n_out=23)

    # create a list of all model parameters
    self.paramList = [self.layer3.params, self.layer2.params, self.layer1a.params, self.layer0a.params]
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
        save_file = open(networkfile, 'rb')
        for p in self.params:
          p.set_value(cPickle.load(save_file), borrow=False)
        save_file.close()


  def classify(self, candidateAndFillerAndOffsetList, slot):
    ##############
    # TEST MODEL #
    ##############

    logger.info('... testing')

    index = T.lscalar()  # index to a [mini]batch

    if self.gotNetwork == 0:
      return []

    inputMatrixDev_a, inputMatrixDev_b, inputMatrixDev_c, lengthListDev_a, lengthListDev_b, lengthListDev_c, inputFeaturesDev, _ = getInput(candidateAndFillerAndOffsetList, self.representationsize, self.contextsize, self.filtersize, self.wordvectors, self.vectorsize)
    # create input matrix and save them in valid_set

    slot2types, binarizer, numTypes = getSlot2Types()

    yner1Dev = []
    yner2Dev = []
    type1bin = binarizer.transform([slot2types[slot][0]])
    type2bin = binarizer.transform([slot2types[slot][1]])
    
    dt = theano.config.floatX

    for item in range(len(inputMatrixDev_a)):
      yner1Dev.append(type1bin)
      yner2Dev.append(type2bin)
    yner1DevNumpy = numpy.array(yner1Dev, dtype = numpy.int32)
    yner2DevNumpy = numpy.array(yner2Dev, dtype = numpy.int32)

    valid_set_xa = theano.shared(numpy.matrix(inputMatrixDev_a, dtype = dt))
    valid_set_xb = theano.shared(numpy.matrix(inputMatrixDev_b, dtype = dt))
    valid_set_xc = theano.shared(numpy.matrix(inputMatrixDev_c, dtype = dt))
    valid_mlp = theano.shared(numpy.matrix(inputFeaturesDev, dtype = dt))
    valid_yner1 = theano.shared(yner1DevNumpy.reshape(yner1DevNumpy.shape[0], yner1DevNumpy.shape[2]))
    valid_yner2 = theano.shared(yner2DevNumpy.reshape(yner2DevNumpy.shape[0], yner2DevNumpy.shape[2]))

    # compute number of minibatches for testing
    n_valid_batches = valid_set_xa.get_value(borrow=True).shape[0]
    n_valid_batches /= self.batch_size

    input_dict = {}
    input_dict[self.xa] = valid_set_xa[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.xb] = valid_set_xb[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.xc] = valid_set_xc[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.yNER1] = valid_yner1[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.yNER2] = valid_yner2[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.additionalFeatures] = valid_mlp[index * self.batch_size: (index + 1) * self.batch_size]

    test_model_confidence = theano.function([index], self.layer3.results(), givens = input_dict)

    resultList = [test_model_confidence(i) for i in xrange(n_valid_batches)]

    return resultList
