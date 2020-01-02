#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
import os
import numpy
from utils import readConfig, readWordvectors, getInput
import cPickle

import theano
import theano.tensor as T
from layers import LogisticRegression, LeNetConvPoolLayer, HiddenLayer, CRF

from sklearn.preprocessing import MultiLabelBinarizer


import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


class CNN:

  def __init__(self, configfile, train = False):

    self.slotList = ["N", "per:age","per:alternate_names","per:children","per:cause_of_death","per:date_of_birth","per:date_of_death","per:employee_or_member_of","per:location_of_birth","per:location_of_death","per:locations_of_residence","per:origin","per:schools_attended", "per:siblings", "per:spouse", "per:title", "org:alternate_names", "org:date_founded", "org:founded_by", "org:location_of_headquarters", "org:members", "org:parents", "org:top_members_employees"]

    typeList = ["O", "PERSON", "LOCATION", "ORGANIZATION", "DATE", "NUMBER"]

    self.config = readConfig(configfile)

    self.addInputSize = 1
    logger.info("additional mlp input")

    wordvectorfile = self.config["wordvectors"]
    logger.info("wordvectorfile " + wordvectorfile)
    networkfile = self.config["net"]
    logger.info("networkfile " + networkfile)
    hiddenunits = int(self.config["hidden"])
    logger.info("hidden units " + str(hiddenunits))
    hiddenunitsNer = hiddenunits
    if "hiddenunitsNER" in self.config:
      hiddenunitsNer = int(self.config["hiddenunitsNER"])
    representationsizeNER = 50
    if "representationsizeNER" in self.config:
      representationsizeNER = int(self.config["representationsizeNER"])
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

    rng = numpy.random.RandomState(23455) # not relevant, parameters will be overwritten by stored model anyways
    if train:
      seed = rng.get_state()[1][0]
      logger.info("seed: " + str(seed))

    numSFclasses = 23
    numNERclasses = 6

    # allocate symbolic variables for the data
    self.index = T.lscalar()  # index to a [mini]batch
    self.xa = T.matrix('xa')   # left context
    self.xb = T.matrix('xb')   # middle context
    self.xc = T.matrix('xc')   # right context
    self.y = T.imatrix('y')    # label (only present in training)
    self.yNER1 = T.imatrix('yNER1') # label for first entity (only present in training)
    self.yNER2 = T.imatrix('yNER2') # label for second entity (only present in training)
    ishape = [self.representationsize, self.contextsize]  # this is the size of context matrizes

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    logger.info('... building the model')

    # Reshape input matrix to be compatible with LeNetConvPoolLayer
    layer0a_input = self.xa.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0b_input = self.xb.reshape((self.batch_size, 1, ishape[0], ishape[1]))
    layer0c_input = self.xc.reshape((self.batch_size, 1, ishape[0], ishape[1]))

    y_reshaped = self.y.reshape((self.batch_size, 1))
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

    layer0aflattened = self.layer0a.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0bflattened = self.layer0b.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0cflattened = self.layer0c.output.flatten(2).reshape((self.batch_size, nkerns[0] * sizeAfterPooling))
    layer0outputSF = T.concatenate([layer0aflattened, layer0bflattened, layer0cflattened], axis = 1)
    layer0outputSFsize = 3 * (nkerns[0] * sizeAfterPooling)

    layer0outputNER1 = T.concatenate([layer0aflattened, layer0bflattened], axis = 1)
    layer0outputNER2 = T.concatenate([layer0bflattened, layer0cflattened], axis = 1)
    layer0outputNERsize = 2 * (nkerns[0] * sizeAfterPooling)

    layer2ner1 = HiddenLayer(rng, input=layer0outputNER1, n_in=layer0outputNERsize,
                           n_out=hiddenunitsNer, activation=T.tanh)
    layer2ner2 = HiddenLayer(rng, input=layer0outputNER2, n_in=layer0outputNERsize,
                           n_out=hiddenunitsNer, activation=T.tanh, W = layer2ner1.W, b = layer2ner1.b)
  
    # concatenate additional features to sentence representation
    self.additionalFeatures = T.matrix('additionalFeatures')
    self.additionalFeatsShaped = self.additionalFeatures.reshape((self.batch_size, 1))
  
    layer2SFinput = T.concatenate([layer0outputSF, self.additionalFeatsShaped], axis = 1)
    layer2SFinputSize = layer0outputSFsize + self.addInputSize
  
    layer2SF = HiddenLayer(rng, input=layer2SFinput, n_in=layer2SFinputSize,
                           n_out=hiddenunits, activation=T.tanh)
  
    # classify the values of the fully-connected sigmoidal layer
    layer3rel = LogisticRegression(input=layer2SF.output, n_in=hiddenunits, n_out=numSFclasses)
    layer3et = LogisticRegression(input=layer2ner1.output, n_in=hiddenunitsNer, n_out=numNERclasses)
  
    scoresForR1 = layer3rel.getScores(layer2SF.output)
    scoresForE1 = layer3et.getScores(layer2ner1.output)
    scoresForE2 = layer3et.getScores(layer2ner2.output)
  
    self.crfLayer = CRF(numClasses = numSFclasses + numNERclasses, rng = rng, batchsizeVar = self.batch_size, sequenceLength = 3)
  
    scores = T.zeros((self.batch_size, 3, numSFclasses + numNERclasses))
    scores = T.set_subtensor(scores[:,0,numSFclasses:], scoresForE1)
    scores = T.set_subtensor(scores[:,1,:numSFclasses], scoresForR1)
    scores = T.set_subtensor(scores[:,2,numSFclasses:], scoresForE2)
    self.scores = scores

    self.y_conc = T.concatenate([yNER1reshaped + numSFclasses, y_reshaped, yNER2reshaped + numSFclasses], axis = 1)

    # create a list of all model parameters
    self.paramList = [self.crfLayer.params, layer3rel.params, layer3et.params, layer2SF.params, layer2ner1.params, self.layer0a.params]
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

    self.relation_scores_global = self.crfLayer.getProbForClass(self.scores, numSFclasses)
    self.predictions_global = self.crfLayer.getPrediction(self.scores)

  def classify(self, candidateAndFillerAndOffsetList, slot):
    ##############
    # TEST MODEL #
    ##############

    slotIndex = self.slotList.index(slot)
    logger.info('... testing')
    logger.info("boosting CNN score if best path goes through class")

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

    input_dict = {}
    input_dict[self.xa] = valid_set_xa[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.xb] = valid_set_xb[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.xc] = valid_set_xc[index * self.batch_size: (index + 1) * self.batch_size]
    input_dict[self.additionalFeatures] = valid_mlp[index * self.batch_size: (index + 1) * self.batch_size]
    test_model = theano.function([index], self.relation_scores_global, givens = input_dict)
    predictions_model = theano.function([index], self.predictions_global, givens = input_dict)

    resultList = [test_model(i) for i in xrange(n_valid_batches)]
    predictionsList = [predictions_model(i)[0][0,2] for i in xrange(n_valid_batches)]
    confidenceList = []
    for r in range(len(resultList)):
      predclass = predictionsList[r]
      conf = resultList[r][0,slotIndex]
      if predclass == slotIndex:
        conf = max(1.0, conf + 0.75) # boost confidence if best path goes through class
      confidenceList.append(conf)
      
    return confidenceList
