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
import copy

class Output():

  def setResults(self,myResults):
    self.myResults = copy.deepcopy(myResults)

  def writeResults(self, resultFile):

    self.logger.info("writing results to file " + resultFile)

    out = open(resultFile, 'a')
    for r in self.myResults:
      out.write(r + "\n")
    out.close()

  def __init__(self, loggerMain):
    self.myResults = []
    self.logger = loggerMain.getChild(__name__)
