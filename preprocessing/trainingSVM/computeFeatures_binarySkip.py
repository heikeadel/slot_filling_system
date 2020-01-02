#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
from scipy.io import mmwrite
from sklearn.feature_extraction.text import CountVectorizer
import gzip
from scipy import sparse
import numpy
import time
import re

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if len(sys.argv) < 3 or len(sys.argv) > 4:
  logger.error("please pass the input text file and the desired output filename as parameter. You can also pass a vocabulary file as input")
  exit()

infile = sys.argv[1]
outfile = sys.argv[2]

vocabIsSet = False
skipVocab = {}
if len(sys.argv) == 4:
  vocabfile = sys.argv[3]
  vocabIsSet = True
  vocab = []
  f = open(vocabfile, 'r')
  for line in f:
    line = line.strip()
    vocab.append(line)
  f.close()
  f = open(vocabfile + ".skip", 'r')
  for index, line in enumerate(f):
    line = line.strip()
    skipVocab[line] = index
  f.close()

f = open(infile, 'r')
corpus = []
leftContexts = []
middleContexts = []
rightContexts = []
labels = []
flagRows = []
flagCols = []
flagValues = []
skipNValues = []
skipNRows = []
skipNCols = []
index = 0
curSkipVocabIndex = 0
# file format:
# +/- : slot : query entity : filler : proof sentence
# alternative line separator: '::' instead of ':'
for line in f:
  line = line.strip()
  if re.search(r'^\S+ \: ', line):
    parts = line.split(" : ")
  elif re.search(r'^\S+ \:\: ', line):
    parts = line.split(" :: ")
  else:
    logger.error("unknown format in line " + line)
    continue
  if len(parts) < 5:
    logger.error("wrong number of columns in line " + line)
    continue
  example = " : ".join(parts[4:])
  label = parts[0]
  if label == '+':
    labelInt = '1'
  elif label == '-':
    labelInt = '-1'
  else:
    print "ERROR: unknown label: " + label
  labels.append(labelInt)
  corpus.append(example)
  exampleList = example.split()
  # improved index computation for splitting:
  # get all occurrences of <name> and <filler> and split where they are closest to each other
  # (idea: no <name> or <filler> tag in the middle context: keep middle context clean)
  fillerIndices = [i for i, x in enumerate(exampleList) if x == "<filler>"]
  nameIndices = [i for i, x in enumerate(exampleList) if x == "<name>"]
  fillerInd = -1
  nameInd = -1
  distanceNameFiller = len(exampleList)
  for fi in fillerIndices:
    for ni in nameIndices:
      distance = abs(ni - fi)
      if distance < distanceNameFiller:
        distanceNameFiller = distance
        nameInd = ni
        fillerInd = fi
  minInd = 0
  maxInd = 0
  if fillerInd < nameInd:
    nameBeforeFiller = 0
    minInd = fillerInd
    maxInd = nameInd
  else:
    nameBeforeFiller = 1
    maxInd = fillerInd
    minInd = nameInd
  flagRows.append(index)
  flagCols.append(0)
  flagValues.append(nameBeforeFiller)
  leftC = " ".join(exampleList[:minInd])
  middleC = " ".join(exampleList[minInd + 1:maxInd])
  rightC = " ".join(exampleList[maxInd + 1:])
  leftContexts.append(leftC)
  middleContexts.append(middleC)
  rightContexts.append(rightC)

  mcList = exampleList[minInd + 1:maxInd]
  foundSkipNgram = False
  for n in range(3,5):
    for i in range(0,len(mcList) + 1 - n):
      curContext = []
      for j in range(0, n):
        if j == 0 or j == n-1:
          curContext.append(mcList[i+j])
      curContextString = " ".join(curContext)
      if curContextString in skipVocab:
        curIndex = skipVocab[curContextString]
        skipNRows.append(index)
        skipNCols.append(curIndex)
        skipNValues.append(1)
        foundSkipNgram = True
      else:
        if vocabIsSet == False:
          curIndex = curSkipVocabIndex
          skipVocab[curContextString] = curSkipVocabIndex
          curSkipVocabIndex += 1
          skipNRows.append(index)
          skipNCols.append(curIndex)
          skipNValues.append(1)
          foundSkipNgram = True

  if foundSkipNgram == False:
    skipNRows.append(index)
    skipNCols.append(0)
    skipNValues.append(0)

  index += 1
f.close()

flagMatrix = sparse.csr_matrix((numpy.array(flagValues), (numpy.array(flagRows), numpy.array(flagCols))), shape = (flagRows[-1] + 1, 1))
skipNCounts = sparse.csr_matrix((numpy.array(skipNValues), (numpy.array(skipNRows), numpy.array(skipNCols))), shape = (skipNRows[-1] + 1, len(skipVocab.keys())))

# normal ngrams:
if vocabIsSet:
  ngram_vectorizer = CountVectorizer(
        ngram_range=(1,3),
        lowercase=False,
        binary=True,
        token_pattern=u'[^ ]+',
        vocabulary=vocab
  )
else:
  ngram_vectorizer = CountVectorizer(
        ngram_range=(1,3),
        lowercase=False,
        binary=True,
        token_pattern=u'[^ ]+'
  )
  ngram_vectorizer.fit(corpus)

counts = ngram_vectorizer.transform(corpus)
leftCounts = ngram_vectorizer.transform(leftContexts)
middleCounts = ngram_vectorizer.transform(middleContexts)
rightCounts = ngram_vectorizer.transform(rightContexts)

# stack all features:
counts = sparse.hstack((flagMatrix, counts))
counts = sparse.hstack((counts, leftCounts))
counts = sparse.hstack((counts, middleCounts))
counts = sparse.hstack((counts, rightCounts))
counts = sparse.hstack((counts, skipNCounts))

mmwrite(outfile, counts)

labelsOut = open(outfile + ".labels", 'w')
labelsOut.write('\n'.join(labels))
labelsOut.close()

if not vocabIsSet:
  vocab = ngram_vectorizer.get_feature_names()
  vocabUnicode = [v.encode('utf8') for v in vocab]
  vocabOut = open(outfile + ".vocab", 'w')
  vocabOut.write('\n'.join(vocabUnicode))
  vocabOut.close()
  vocabSkip = sorted(skipVocab.items(), key=lambda x:x[1])
  vocabSkipWords = [v[0] for v in vocabSkip]
  vocabOut = open(outfile + ".vocab.skip", 'w')
  vocabOut.write('\n'.join(vocabSkipWords))
  vocabOut.close()
