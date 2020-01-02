#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################
# script for training multiclass CNNs for slot filling
# 2016
#####

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../cnnScripts'))

import cPickle
import numpy
import theano
import theano.tensor as T
from utils import readConfig, getInput
from testCNN_multiClass import CNN
from utils_training import getFScoreMultiClass, sgd_updates

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if len(sys.argv) != 2:
  logging.error("please pass the config file for the multiclass CNN as parameter")
  exit()

configfile = sys.argv[1]
config = readConfig(configfile)
trainfile = config["train"]
logger.info("trainfile " + trainfile)
devfile = config["dev"]
logger.info("devfile " + devfile)
wordvectorfile = config["wordvectors"]
networkfile = config["net"]
logger.info("networkfile " + networkfile)
learning_rate = float(config["lrate"])
logger.info("learning rate " + str(learning_rate))
batch_size = int(config["batchsize"])
logger.info("batch size " + str(batch_size))
myLambda1 = 0
if "lambda1" in config:
  myLambda1 = float(config["lambda1"])
myLambda2 = 0
if "lambda2" in config:
  myLambda2 = float(config["lambda2"])
logger.info("lambda1 " + str(myLambda1))
logger.info("lambda2 " + str(myLambda2))

# load model architecture and word vectors etc
multiCNN = CNN(configfile, train = True)

trainfilehandle = open(trainfile)
inputMatrixTrain_a, inputMatrixTrain_b, inputMatrixTrain_c, length_a, length_b, length_c, inputFeaturesTrain, resultVectorTrain = getInput(trainfilehandle, multiCNN.representationsize, multiCNN.contextsize, multiCNN.filtersize, multiCNN.wordvectors, multiCNN.vectorsize)
trainfilehandle.close()
devfilehandle = open(devfile)
inputMatrixDev_a, inputMatrixDev_b, inputMatrixDev_c, length_a, length_b, length_c, inputFeaturesDev, resultVectorDev = getInput(devfilehandle, multiCNN.representationsize, multiCNN.contextsize, multiCNN.filtersize, multiCNN.wordvectors, multiCNN.vectorsize)
devfilehandle.close()

dt = theano.config.floatX

train_set_xa = theano.shared(numpy.matrix(inputMatrixTrain_a, dtype = dt), borrow=True)
valid_set_xa = theano.shared(numpy.matrix(inputMatrixDev_a, dtype = dt), borrow=True)
train_set_xb = theano.shared(numpy.matrix(inputMatrixTrain_b, dtype = dt), borrow=True)
valid_set_xb = theano.shared(numpy.matrix(inputMatrixDev_b, dtype = dt), borrow=True)
train_set_xc = theano.shared(numpy.matrix(inputMatrixTrain_c, dtype = dt), borrow=True)
valid_set_xc = theano.shared(numpy.matrix(inputMatrixDev_c, dtype = dt), borrow=True)
train_set_y = theano.shared(numpy.matrix(resultVectorTrain, dtype = numpy.dtype(numpy.int32)).reshape(len(resultVectorTrain), 1), borrow=True)
train_mlp = theano.shared(numpy.matrix(inputFeaturesTrain, dtype = dt), borrow=True)
valid_set_y = theano.shared(numpy.array(resultVectorDev, dtype = numpy.dtype(numpy.int32)), borrow=True)
valid_mlp = theano.shared(numpy.matrix(inputFeaturesDev, dtype = dt), borrow=True)

index = T.lscalar()  # index to a [mini]batch
lr = learning_rate

params = multiCNN.params
cost = multiCNN.layer3.negative_log_likelihood(multiCNN.y_reshaped)

# add regularization
reg2 = 0
reg1 = 0
for p in multiCNN.paramList:
  reg2 += T.sum(p[0] ** 2)
  reg1 += T.sum(abs(p[0]))
cost += myLambda2 * reg2
cost += myLambda1 * reg1

grads = T.grad(cost, params)
updates = sgd_updates(params, cost, lr)

# define theano functions
start = index * batch_size
stop = (index + 1) * batch_size
train = theano.function([index], cost, updates = updates, givens = {
                multiCNN.xa: train_set_xa[start : stop],
                multiCNN.xb: train_set_xb[start : stop],
                multiCNN.xc: train_set_xc[start : stop],
                multiCNN.additionalFeatures: train_mlp[start : stop],
                multiCNN.y : train_set_y[start : stop]})
validate = theano.function([index], multiCNN.layer3.results(), givens = {
                multiCNN.xa: valid_set_xa[start : stop],
                multiCNN.xb: valid_set_xb[start : stop],
                multiCNN.xc: valid_set_xc[start : stop],
                multiCNN.additionalFeatures: valid_mlp[start : stop]})

logger.info("... training")
# train model
n_epochs=15
best_params = []
best_fscore = -1
epoch = 0

n_valid_batches = inputMatrixDev_a.shape[0] / batch_size

numpy.random.seed(987654)

maxNumPerEpoch = 50000 # change according to computing ressources
numPerEpoch = min(inputMatrixTrain_a.shape[0], maxNumPerEpoch)
n_train_batches = numPerEpoch / batch_size

while epoch < n_epochs:
        logger.info('epoch = ' + str(epoch))
        epoch = epoch + 1

        # shuffling data for batch
        randomIndices = numpy.random.permutation(inputMatrixTrain_a.shape[0])
        randomIndicesThis = randomIndices[0:numPerEpoch]
        train_set_xa.set_value(numpy.matrix(inputMatrixTrain_a[randomIndicesThis], dtype = dt), borrow=True)
        train_set_xb.set_value(numpy.matrix(inputMatrixTrain_b[randomIndicesThis], dtype = dt), borrow=True)
        train_set_xc.set_value(numpy.matrix(inputMatrixTrain_c[randomIndicesThis], dtype = dt), borrow=True)
        train_mlp.set_value(numpy.matrix(inputFeaturesTrain[randomIndicesThis], dtype = dt), borrow=True)
        thisResultVectorTrain = []
        for ri in randomIndicesThis:
          thisResultVectorTrain.append(resultVectorTrain[ri])
        train_set_y.set_value(numpy.array(thisResultVectorTrain, dtype = numpy.dtype(numpy.int32)).reshape(len(thisResultVectorTrain), 1), borrow=True)

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                logger.debug('training @ iter = ' + str(iter))

            cost_ij = train(minibatch_index)

        confidence = [validate(i) for i in xrange(n_valid_batches)]
        this_fscore = getFScoreMultiClass(confidence, resultVectorDev, batch_size)

        logger.info('epoch ' + str(epoch) + ", learning_rate " + str(learning_rate) + ", minibatch " + str(minibatch_index + 1) + "/" + str(n_train_batches) + ", validation fscore " + str(this_fscore))

        # if we got the best validation score until now
        if this_fscore > best_fscore:
          # save best validation score and iteration number
          best_fscore = this_fscore
          best_params = []
          for p in multiCNN.params:
            best_params.append(p.get_value(borrow=False))

logger.info('Optimization complete.')
# save best parameters
save_file = open(networkfile, 'wb')
for p in best_params:
  cPickle.dump(p, save_file, -1)

