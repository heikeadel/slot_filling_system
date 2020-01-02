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
import re
import os.path
import os
import io
import gzip

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

def getOriginalCharacter(char):
  result = char
  if char == '-LRB-':
    result = "("
  elif char == '-RRB-':
    result = ")"
  elif char == '-LCB-':
    result = "{"
  elif char == '-RCB-':
    result = "}"
  elif char == '-LSB-':
    result = "["
  elif char == '-RSB-':
    result = "]"
  elif char == '``':
    result = '"'
  elif char == "''":
    result = '"'
  elif char == '`':
    result = "'"
  result = re.sub(ur'\&gt\;', '>', result, re.UNICODE)
  result = re.sub(ur'\&lt\;', '<', result, re.UNICODE)
  return result

def readDocument(docId, docPath):
  docString = ""
  if not os.path.isfile(docPath):
    docPath = re.sub(ur'\.gz', '', docPath, re.UNICODE)
    if not os.path.isfile(docPath):
      logger.error("ERROR: could not find " + docPath)
      return docString
  if re.search(ur'\.gz$', docPath, re.UNICODE):
    g = gzip.GzipFile(docPath)
    f = io.TextIOWrapper(io.BufferedReader(g), encoding='utf-8')
  else:
    f = io.open(docPath, encoding='utf-8')
  readingCorrectDoc = 0
  for line in f:
    if readingCorrectDoc == 1:
      if re.search(ur'^\s*$', line, re.UNICODE):
        continue
      if "</doc>" in line or "</DOC>" in line:
        docString += line
        readingCorrectDoc = 0
        break
      line = re.sub(ur'\xc2\xa0', ' ', line, re.UNICODE)
      line2 = re.sub(ur'\n$', '', line, re.UNICODE)
      docString += line2 + "\n"
    else:
      if "doc id" in line:
        docno = re.sub(ur'^\<doc id\=\"', '', line, re.UNICODE)
        docno = re.sub(ur'\"\>\s*$', '', docno, re.UNICODE)
        if docno == docId:
          docString += line
          readingCorrectDoc = 1
      elif "DOCID" in line:
        docno = re.sub(ur"^\<DOCID\> ", '', line, re.UNICODE)
        docno = re.sub(ur"\s*\<\/DOCID\>\s*$", '', docno, re.UNICODE)
        if docno == docId:
          docString += line
          readingCorrectDoc = 1
      elif "DOC id" in line:
        docno = re.sub(ur'^\<DOC id\=\"', '', line, re.UNICODE)
        docno = re.sub(ur'\" type.*\>\s*$', '', docno, re.UNICODE)
        if docno == docId:
          docString += line
          readingCorrectDoc = 1
      elif "DOC docid" in line:
        docno = re.sub(ur'^\<DOC docid\=\"', '', line, re.UNICODE)
        docno = re.sub(ur'\"\>\s*$', '', docno, re.UNICODE)
        if docno == docId:
          docString += line
          readingCorrectDoc = 1
      elif "</doc>" in line or "</DOC>" in line:
        docString = ""
      elif "<doc" in line or "<DOC" in line:
        docString += line
  f.close()
  return docString

def processOutput(docId):
  taggingPath = TAGGINGPATH # replace with path to store tagged documents
  corefFile = taggingPath + "/" + docId + ".xml"
  if not os.path.isfile(corefFile):
    logger.error("file " + taggingPath + "/" + docId + ".xml" + " does not exist")
    return
  f = io.open(corefFile, encoding='utf-8')
  out = open(taggingPath + "/" + docId + ".prepared", 'w')
  readingCoref = 0
  curSentId = ""
  curTokenId = ""
  curWord = ""
  curPos = ""
  curLemma = ""
  curBegin = ""
  curEnd = ""
  curNormDate = ""
  output = ""
  curChain = []
  curMention = []
  for line in f:
    if readingCoref == 1:
      if "</coreference>" in line:
        if len(curChain) > 0:
          c = curChain[0]
          cOut = str(c[0])
          for cOutIndex in range(1, len(c)):
            cOut += "-" + str(c[cOutIndex])
          out.write(cOut)
          for cIndex in range(1, len(curChain)):
            c = curChain[cIndex]
            cOut = str(c[0])
            for cOutIndex in range(1, len(c)):
              cOut += "-" + str(c[cOutIndex])
            out.write("___" + cOut)
          out.write("\n")
        curChain = []
      elif "<sentence>" in line:
        curSentId = int(re.sub(ur'\s*\<sentence\>(.*?)\<\/sentence\>\s*', '\\1', line, re.UNICODE))
        curMention.append(curSentId)
      elif "<start>" in line:
        curStart = int(re.sub(ur'\s*\<start\>(.*?)\<\/start\>\s*', '\\1', line, re.UNICODE))
        curMention.append(curStart)
      elif "<end>" in line:
        curEnd = int(re.sub(ur'\s*\<end\>(.*?)\<\/end\>\s*', '\\1', line, re.UNICODE))
        curMention.append(curEnd)
      elif "<text>" in line:
        curChain.append(curMention)
        curMention = []
    elif "<sentence id" in line:
      curSentId = re.sub(ur'^\s*\<sentence id\=\"(\d+)\"\>\s*$', '\\1', line, re.UNICODE)
    elif "<token id" in line:
      curTokenId = re.sub(ur'^\s*\<token id\=\"(\d+)\"\>\s*$', '\\1', line, re.UNICODE)
    elif "<lemma>" in line:
      curLemma = re.sub(ur'^\s*\<lemma\>(.*?)\<\/lemma\>\s*$', '\\1', line, re.UNICODE)
      curLemma = getOriginalCharacter(curLemma)
      curLemma = curLemma.encode('utf-8')
      curLemma = re.sub(ur'\xc2\xa0', '-_-', curLemma, re.UNICODE)
      curLemma = curLemma.decode('utf-8')
    elif "<CharacterOffsetBegin>" in line:
      curBegin = re.sub(ur'^\s*\<CharacterOffsetBegin\>(.*?)\<\/CharacterOffsetBegin\>\s*$', '\\1', line, re.UNICODE)
    elif "<CharacterOffsetEnd>" in line:
      curEnd = re.sub(ur'^\s*\<CharacterOffsetEnd\>(.*?)\<\/CharacterOffsetEnd\>\s*$', '\\1', line, re.UNICODE)
    elif "<word>" in line:
      newLine = re.sub(ur'^\s*\<word\>(.*?)\<\/word\>\s*$', '\\1', line, re.UNICODE)
      curWord = getOriginalCharacter(newLine)
      curWord = curWord.encode('utf-8')
      curWord = re.sub(ur'\xc2\xa0', '-_-', curWord, re.UNICODE)
      curWord = curWord.decode('utf-8')
    elif "<POS>" in line:
      curPos = re.sub(ur'^\s*\<POS\>(.*?)\<\/POS\>\s*$', '\\1', line, re.UNICODE)
    elif "<NER>" in line:
      curNer = re.sub(ur'^\s*\<NER\>(.*?)\<\/NER\>\s*$', '\\1', line, re.UNICODE)
      output += curSentId + "-" + curTokenId + "___" + curWord + "___" + curBegin + "-" + curEnd + "___" + curLemma + "___" + curPos + "___" + curNer
    elif "<NormalizedNER>" in line:
      curNormDate = re.sub(ur'^\s*\<NormalizedNER\>(.*?)\<\/NormalizedNER\>\s*$', '\\1', line, re.UNICODE)
    elif 'type="DATE"' in line:
      output += "__" + curNormDate
    elif "</token>" in line:
      output += " "
    elif "</tokens>" in line:
      out.write(output + "\n")
      output = ""
    elif "</sentences>" in line:
      out.write("<coreference>" + "\n")
      readingCoref = 1
  f.close()
  out.close()

def tagDocument(docId, docPath, isBolt = 0):
  taggingPath = TAGGINGPATH # replace with path to store tagged documents
  if os.path.isfile(taggingPath + "/" + docId):
    logger.debug(docId + " has already been tagged")
  else:
    out = open(taggingPath + "/" + docId, 'w')
    docString = readDocument(docId, docPath)
    numLines = len(docString.split('\n'))
    out.write(docString)
    out.close()
    # call tagger
    pwd = os.getcwd()
    os.chdir("PATH/stanford-corenlp-full-2014-01-04") # replace with path to Stanford CoreNLP
    if numLines < 1000:
      if isBolt == 1:
        os.system("java -cp stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -ssplit.newlineIsSentenceBreak always -file " + taggingPath +"/" + docId + " -outputDirectory " + taggingPath)
      else:
        os.system("java -cp stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file " + taggingPath +"/" + docId + " -outputDirectory " + taggingPath)
    else:
      if isBolt == 1:
        os.system("java -cp stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.newlineIsSentenceBreak always -file " + taggingPath +"/" + docId + " -outputDirectory " + taggingPath)
      else:
        os.system("java -cp stanford-corenlp-3.3.1.jar:stanford-corenlp-3.3.1-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -file " + taggingPath +"/" + docId + " -outputDirectory " + taggingPath)
    os.chdir(pwd)
  # process output
  processOutput(docId)
  # save storage space
  if os.path.isfile(taggingPath + "/" + docId):
    os.remove(taggingPath + "/" + docId)
  if os.path.isfile(taggingPath + "/" + docId + ".xml"):
    os.remove(taggingPath + "/" + docId + ".xml")
  if os.path.isfile(taggingPath + "/" + docId + ".prepared"):
    os.system("gzip -f " + taggingPath + "/" + docId + ".prepared")
