#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

from __future__ import unicode_literals
import codecs, sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)
import getOffsets
import readNerAndCoref
from utilities import compareNamesImproved
import re

class Document:

  def preprocessDoc(self, listOfTokenizedAlias, isPer, isOrg, isLoc, isPersonFiller = 1):
    # do coreference resolution and named entity tagging
    wordTags = readNerAndCoref.nerAndCoref(listOfTokenizedAlias, self.docId, self.docPath, isPer, isOrg, isLoc, self.doCoref, isPersonFiller = isPersonFiller)
    
    if len(wordTags) == 0:
      self.logger.error("error in ner- and coref- alignment")
      return -1

    # results:
    self.sentences = wordTags[0]
    self.nerInSentences = wordTags[1]
    self.offsets = wordTags[2]
    self.nameOffsets = list(set(wordTags[3]))
    self.origNameOffsets = wordTags[4]
    self.posInSentences = wordTags[5]
    self.lemmasInSentence = wordTags[6]
    self.offsets2normDate = wordTags[7]
    self.additionalNameOffsets = wordTags[8]
    self.spellingVariationsOffsets = wordTags[9]
    self.additionalCorefFillers = wordTags[10]

    # adjust offsets: CoreNLP does not count spaces at the beginning of lines, but TAC does
    newOffsets, newOffset2NormDate, newAdditionalNameOffsets, newNameOffsets, newOrigNameOffsets, newAdditionalCorefFillers = getOffsets.correctOffsets(self.docId, self.docPath, self.sentences, self.offsets, self.offsets2normDate, self.additionalNameOffsets, self.nameOffsets, self.origNameOffsets, self.additionalCorefFillers)

    if len(newOffsets) == 0: # should not happen
      self.logger.error("length offsets per line old: " + str(len(self.offsets)) + ", and new: " + str(len(newOffsets)))
    else:
      self.offsets = newOffsets
      self.offsets2normDate = newOffset2NormDate
      self.additionalNameOffsets = newAdditionalNameOffsets
      self.nameOffsets = newNameOffsets
      self.origNameOffsets = newOrigNameOffsets
      self.additionalCorefFillers = newAdditionalCorefFillers

  def isFullNameInDoc(self, listOfAlias):
    # open document: get words and their offsets
    sentencesOffsets = getOffsets.getOffsets(self.docId, self.docPath)
    if len(sentencesOffsets) == 0:
      self.logger.error("error in parsing document")
      return 0
    thisSentences = " ".join([so[0] for so in sentencesOffsets])
    thisOffsets = " ".join([str(so[1]) for so in sentencesOffsets])
    if self.docId in self.doc2wordsAndOffsets:
      # document has already been processed
      return 0
    else:
      self.doc2wordsAndOffsets[self.docId] = []
      for so in zip(thisSentences.split(), thisOffsets.split()):
        self.doc2wordsAndOffsets[self.docId].append(so)

    # test whether name is in document!
    sentenceString = thisSentences.decode('utf-8')
    sentenceString = re.sub(r'\s+', ' ', sentenceString)
    sentenceString = re.sub(r'(\W+)', ' \\1 ', sentenceString)
    sentenceString = re.sub(r'\s+', ' ', sentenceString)
    sentenceString = sentenceString.encode('utf-8')
    foundName = 0
    for myAlias in listOfAlias:
      myA = myAlias.decode('utf-8')
      myA = re.sub(r'(\W+)', ' \\1 ', myA)
      myA = re.sub(r'\s+', ' ', myA)
      myA = myA.encode('utf-8')
      foundList = compareNamesImproved(myA, sentenceString)
      if len(foundList) > 0:
        foundName = 1
        break
      else:
        foundList = compareNamesImproved(myAlias, sentenceString)
        if len(foundList) > 0:
          foundName = 1
          break
    return foundName
    

  def setDocId(self, d):
    self.docId = d

  def setDocPath(self, docPath):
    self.docPath = docPath

  def resetDocInfos(self):
    self.docId = ""
    self.docPath = ""
    self.doc2wordsAndOffsets = {}

  def __init__(self, doCoref, loggerMain):
    self.docId = ""
    self.docPath = ""
    self.doc2wordsAndOffsets = {}
    self.spellingVariationsOffsets = {}
    self.doCoref = doCoref

    self.logger = loggerMain.getChild(__name__)
