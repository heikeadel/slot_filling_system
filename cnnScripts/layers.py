#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

#####
# Description: Implementation of network layers
# Date: 2015-2016
#
# References:
# Code for HiddenLayer and LogisticRegression: based on Theano tutorials
# Code for CRFLayer: based on https://github.com/glample/tagger [Lample et al. 2016]. See also: http://cistern.cis.lmu.de/globalNormalization/
#####

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN


class HiddenLayer(object):
    # Code based on Theano tutorial (http://deeplearning.net/tutorial/code/mlp.py)
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=""):
        """
        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if name != "":
          prefix = name
        else:
          prefix = "mlp_"

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

#########################################################################################

class LogisticRegression(object):
    # Code based on Theano Tutorial (http://deeplearning.net/tutorial/code/logistic_sgd.py)
    def __init__(self, input, n_in, n_out, W = None, b = None, rng = None, dropout_rate = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        if W == None:
          # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
          self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='softmax_W', borrow=True)
        else:
          self.W = W

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='softmax_b', borrow=True)
        else:
          self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x_sigmoid = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def results(self):
      """Return the predicted class, the maximum probability and probabilities for all classes.
      """
      return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]


    def getScores(self, curInput):
      """Return the scores before softmax.

      :type curInput: theano.tensor.TensorType
      :param curInput: the input vector to the softmax layer
      """
      predictions = T.dot(curInput, self.W) + self.b
      return predictions


    def cross_entropy(self, y):
      """Return the cross entropy cost of the model.

      :type y: theano.tensor.TensorType
      :param y: corresponds to a vector that gives for each example the correct label
      """
      pred = T.clip(self.p_y_given_x_sigmoid, 0.0001, 0.9999) # for log
      cost = -T.sum(y * T.log(pred) + (1.0 - y) * T.log(1.0 - pred), axis = 1) # sum over all labels
      return T.mean(cost, axis = 0) # mean over samples

####################################################################################

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def preparePooling(self, conv_out):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling
      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      neighborsArgSorted = neighborsArgSorted
      return neighborsForPooling, neighborsArgSorted

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling, neighborsArgSorted = self.preparePooling(conv_out)
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape = None, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        self.conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        k = poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)

        self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

###################################################################################

class CRF:

    # Code based on (https://github.com/glample/tagger) [Lample et al. 2016]
    def log_sum_exp(self, x, axis=None):
      """
      Sum probabilities in the log-space.
      """
      xmax = x.max(axis=axis, keepdims=True)
      xmax_ = x.max(axis=axis)
      return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def recurrence(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        return self.log_sum_exp(previous + obs + self.transitions.dimshuffle('x', 0, 1), axis=1)

    def recurrence_viterbi(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        scores = previous + obs + self.transitions.dimshuffle('x', 0, 1)
        out = scores.max(axis=1)
        return out

    def recurrence_viterbi_returnBest(self, obs, previous):
        previous = previous.dimshuffle(0, 1, 'x')
        obs = obs.dimshuffle(0, 'x', 1)
        scores = previous + obs + self.transitions.dimshuffle('x', 0, 1)
        out = scores.max(axis=1)
        out2 = scores.argmax(axis=1)
        return out, out2

    def forward(self, observations, viterbi=False, return_alpha=False, return_best_sequence=False):
      """
      Inputs:
        - observations, sequence of shape (batch_size, n_steps, n_classes)
      Probabilities must be given in the log space.
      Compute alpha, matrix of size (batch_size, n_steps, n_classes), such that
      alpha[:, i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
      Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
      """
      assert not return_best_sequence or (viterbi and not return_alpha)

      def recurrence_bestSequence(b):
        sequence_b, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][b][-1]), 'int32'),
            sequences=T.cast(alpha[1][b,::-1], 'int32')
        )
        return sequence_b

      initial = observations[:,0]

      if viterbi:
        if return_best_sequence:
          alpha, _ = theano.scan(
            fn=self.recurrence_viterbi_returnBest,
            outputs_info=(initial, None),
            sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
          )
          alpha[0] = alpha[0].dimshuffle(1,0,2) # shuffle back
          alpha[1] = alpha[1].dimshuffle(1,0,2)
        else:
          alpha, _ = theano.scan(
            fn=self.recurrence_viterbi,
            outputs_info=initial,
            sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
          )
          alpha = alpha.dimshuffle(1,0,2) # shuffle back
      else:
        alpha, _ = theano.scan(
          fn=self.recurrence,
          outputs_info=initial,
          sequences=[observations[:,1:].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
        )
        alpha = alpha.dimshuffle(1,0,2) # shuffle back

      if return_alpha:
        return alpha
      elif return_best_sequence: # TODO: is there a better way than 2 nested scans?
        batchsizeVar = alpha[0].shape[0]
        sequence, _ = theano.scan(
           fn=recurrence_bestSequence,
           outputs_info = None,
           sequences=T.arange(batchsizeVar)
        )
        sequence = T.concatenate([sequence[:,::-1], T.argmax(alpha[0][:,-1], axis = 1).reshape((batchsizeVar, 1))], axis = 1)
        return sequence, alpha[0]
      else:
        if viterbi:
          return alpha[:,-1,:].max(axis=1)
        else:
          return self.log_sum_exp(alpha[:,-1,:], axis=1)

    def backward(self, observations):
      batchsizeVar = observations.shape[0]
      numCl = observations.shape[2]
      initial = T.ones(shape = (batchsizeVar,numCl), dtype = theano.config.floatX)
      beta, _ = theano.scan(
        fn=self.recurrence_viterbi,
        outputs_info=initial,
        sequences=[observations[:,:-1][:,::-1].dimshuffle(1,0,2)] # shuffle to get a sequence over time, not over batches
      )
      beta = beta.dimshuffle(1,0,2) # shuffle back
      return beta

    def __init__(self, numClasses, rng, batchsizeVar, sequenceLength = 3):
      self.numClasses = numClasses

      shape_transitions = (numClasses + 2, numClasses + 2) # +2 because of start id and end id
      drange = numpy.sqrt(6.0 / numpy.sum(shape_transitions))
      self.transitions = theano.shared(value = numpy.asarray(rng.uniform(low = -drange, high = drange, size = shape_transitions), dtype = theano.config.floatX), name = 'transitions')

      self.small = -1000 # log for very small probability
      b_s = numpy.array([[self.small] * numClasses + [0, self.small]]).astype(theano.config.floatX)
      e_s = numpy.array([[self.small] * numClasses + [self.small, 0]]).astype(theano.config.floatX)
      self.b_s_theano = theano.shared(value = b_s).dimshuffle('x', 0, 1)
      self.e_s_theano = theano.shared(value = e_s).dimshuffle('x', 0, 1)

      self.b_s_theano = self.b_s_theano.repeat(batchsizeVar, axis = 0)
      self.e_s_theano = self.e_s_theano.repeat(batchsizeVar, axis = 0)

      self.s_len = sequenceLength

      self.debug1 = self.e_s_theano

      self.params = [self.transitions]

    def getObservations(self, scores):
      batchsizeVar = scores.shape[0]
      observations = T.concatenate([scores, self.small * T.cast(T.ones((batchsizeVar, self.s_len, 2)), theano.config.floatX)], axis = 2)
      observations = T.concatenate([self.b_s_theano, observations, self.e_s_theano], axis = 1)
      return observations

    def getPrediction(self, scores):
      observations = self.getObservations(scores)
      prediction = self.forward(observations, viterbi=True, return_best_sequence=True)
      return prediction

    def getAlpha(self, scores):
      observations = self.getObservations(scores)
      return self.forward(observations, return_alpha=True)

    def getProbForClass(self, scores, numClassesRE):
      batchsizeVar = scores.shape[0]
      observations = self.getObservations(scores)
      alpha = self.forward(observations, return_alpha=True)
      beta = self.backward(observations)

      sum_probs = T.zeros((batchsizeVar, 1))
      classScores = T.zeros((batchsizeVar, numClassesRE))
      for t in range(numClassesRE):
        alpha_prob_relation = alpha[:,1,t].reshape((batchsizeVar, 1))
        beta_prob_relation = beta[:,1,t].reshape((batchsizeVar, 1))
        targetClassScore = alpha_prob_relation + beta_prob_relation
        classScores = T.set_subtensor(classScores[:,t:t+1], targetClassScore)
        sum_probs = sum_probs + targetClassScore
      classScores = classScores / (sum_probs.reshape((batchsizeVar,)).dimshuffle(0,'x'))
      return classScores

    def getCost(self, scores, y_conc):
      batchsizeVar = scores.shape[0]
      observations = self.getObservations(scores)

      # score from classes
      scores_flattened = scores.reshape((scores.shape[0] * scores.shape[1], scores.shape[2]))
      y_flattened = y_conc.flatten(1)

      real_path_score = scores_flattened[T.arange(batchsizeVar * self.s_len), y_flattened]

      real_path_score = real_path_score.reshape((batchsizeVar, self.s_len)).sum(axis = 1)

      # score from transitions
      b_id = theano.shared(value=numpy.array([self.numClasses], dtype=numpy.int32)) # id for begin
      e_id = theano.shared(value=numpy.array([self.numClasses + 1], dtype=numpy.int32)) # id for end
      b_id = b_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)
      e_id = e_id.dimshuffle('x', 0).repeat(batchsizeVar, axis = 0)

      padded_tags_ids = T.concatenate([b_id, y_conc, e_id], axis=1)

      real_path_score2, _ = theano.scan(fn = lambda m: self.transitions[padded_tags_ids[m,T.arange(self.s_len+1)], padded_tags_ids[m,T.arange(self.s_len + 1) + 1]].sum(), sequences = T.arange(batchsizeVar), outputs_info = None)

      real_path_score += real_path_score2
      all_paths_scores = self.forward(observations)
      self.debug1 = real_path_score
      cost = - T.mean(real_path_score - all_paths_scores)
      return cost

