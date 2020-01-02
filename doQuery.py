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
import os
import re
import string
import unicodedata

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

def normalize(thisString):
  return unicodedata.normalize('NFD', thisString).encode('ascii', 'ignore')

########## function for searching two locations in the source corpus (for inference) ##############
def queryTwoLocations(loc1, loc2, terrierDir):
  docNo = 20
  resultingDocs = []
  pwd = os.getcwd()
  os.chdir(terrierDir)
  f = open("var/results/querycounter", 'r')
  number = -1
  for line in f:
    line = line.strip()
    number = int(line)
  f.close()
  number += 1
  queryresultfile = "var/results/Hiemstra_LM0.15_" + str(number) + ".res"
  # write query
  q = open("etc/queries", 'w')
  # AND query
  queryId = 1
  q.write(str(queryId) + ' ')
  for locPart in loc1.split():
    locPart = re.sub(ur'[\.\,\?]', '', locPart, re.UNICODE)
    q.write('+' + locPart + ' ')
  for locPart in loc2.split():
    locPart = re.sub(ur'[\.\,\?]', '', locPart, re.UNICODE)
    q.write('+' + locPart + ' ')
  q.write("\n")
  loc1_norm = normalize(loc1)
  loc2_norm = normalize(loc2)
  if loc1_norm != loc1 or loc2_norm != loc2:
    # AND query with normalized strings
    queryId += 1
    q.write(str(queryId) + ' ')
    for locPart in loc1_norm.split():
      locPart = re.sub(ur'[\.\,\?]', '', locPart, re.UNICODE)
      q.write('+' + locPart + ' ')
    for locPart in loc2_norm.split():
      locPart = re.sub(ur'[\.\,\?]', '', locPart, re.UNICODE)
      q.write('+' + locPart + ' ')
    q.write("\n")
  q.close()
  # perform query
  os.system("./bin/trec_terrier.sh -r -Dtrec.model=Hiemstra_LM -c 0.15")
  # read results
  results = []
  f = open(queryresultfile, 'r')
  for line in f:
    line = line.strip()
    parts = line.split()
    if len(parts) < 6:
      continue
    document = parts[2]
    score = float(parts[4])
    results.append([document, score])
  f.close()
  os.chdir(pwd)

  results = sorted(results, key=lambda x:x[1], reverse=True)

  count = 0
  while count < docNo and count < len(results):
    resultingDocs.append(results[count][0])
    count += 1

  logger.info("IR result for inference: " + str(len(resultingDocs)) + " documents")
  return resultingDocs


########## this function looks for the given name in the source corpus and returns the top matching documents ##########
def query(nameList, docNo, docList, isLoc, terrierDir):

  name = nameList[0]
  resultingDocs = []
  pwd = os.getcwd()
  ## search for documents containing the subjects of the slots
  os.chdir(terrierDir)
  ## get result file name:
  f = open("var/results/querycounter", 'r')
  number = -1
  for line in f:
    line = line.strip()
    number = int(line)
  f.close()
  number += 1
  queryresultfile = "var/results/Hiemstra_LM0.15_" + str(number) + ".res"
  ## write query
  q = open("etc/queries", 'w')

  # AND query of real name
  queryId = 1
  q.write(str(queryId) + ' ')
  for namePart in name.split():
    namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
    q.write('+' + namePart + ' ')
  q.write("\n")

  name_norm = normalize(name)
  if name != name_norm:
    # AND query of normalized real name
    queryId += 1
    q.write(str(queryId) + ' ')
    for namePart in name_norm.split():
      namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
      q.write('+' + namePart + ' ')
    q.write("\n")

  name = name_norm # do the rest with the normalized version of name!

  if isLoc == 0:
    if "-" in name:
      # search without '-'
      queryId += 1
      q.write(str(queryId) + ' ')
      for namePart in name.split():
        namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
        namePartList = namePart.split('-')
        for np in namePartList:
          q.write('+' + np + ' ')
      q.write("\n")

    if re.search(ur'\.\w', name, re.UNICODE):
      # if there is a dot in the middle of the name: split name
      queryId += 1
      q.write(str(queryId) + ' ')
      for namePart in name.split():
        namePart.strip('.')
        namePartList = namePart.split('.')
        for np in namePartList:
          np2 = re.sub(ur'[\.\,\?]', '', np, re.UNICODE)
          q.write('+' + np2 + ' ')
      q.write("\n")

  aliasToSearch = []
  for n in nameList:
    n = normalize(n) # get normalized version
    if string.lower(name) in string.lower(n):
      continue
    aliasToSearch.append(n)

  # AND queries of name aliases
  for myName in aliasToSearch:
    queryId += 1
    myName = myName.strip()
    q.write(str(queryId) + ' ')
    for namePart in myName.split():
      namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
      q.write('+' + namePart + ' ')
    q.write("\n")
    if isLoc == 0:
      if "-" in myName:
        # search without '-'
        queryId += 1
        q.write(str(queryId) + ' ')
        for namePart in myName.split():
          namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
          namePartList = namePart.split('-')
          for np in namePartList:
            q.write('+' + np + ' ')
        q.write("\n")
      if re.search(ur'\.\w', myName, re.UNICODE):
        # if there is a dot in the middle of the name: split name
        queryId += 1
        q.write(str(queryId) + ' ')
        for namePart in myName.split():
          namePart.strip('.')
          namePartList = namePart.split('.')
          for np in namePartList:
            np2 = re.sub(ur'[\.\,\?]', '', np, re.UNICODE)
            q.write('+' + np2 + ' ')
        q.write("\n")

  queryIdAnd = queryId

  if isLoc == 0:
    # OR query of real name
    queryId += 1
    if len(name.split()) > 1:
      q.write(str(queryId) + ' ')
      for namePart in name.split():
        namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
        q.write(namePart + ' ')
      q.write("\n")

    if len(name.split()) > 1 and "-" in name:
      queryId += 1
      q.write(str(queryId) + ' ')
      for namePart in name.split():
        namePart = re.sub(ur'[\.\,\?]', '', namePart, re.UNICODE)
        namePartList = namePart.split('-')
        for np in namePartList:
          q.write(np + ' ')
      q.write("\n")

    if len(name.split()) > 1 and re.search(ur'\.\w', name, re.UNICODE):
      queryId += 1
      q.write(str(queryId) + ' ')
      for namePart in name.split():
        namePart.strip('.')
        namePartList = namePart.split('-')
        for np in namePartList:
          np2 = re.sub(ur'[\.\,\?]', '', np, re.UNICODE)
          q.write(np2 + ' ')
      q.write("\n")

  q.close()

  ## perform query
  os.system("./bin/trec_terrier.sh -r -Dtrec.model=Hiemstra_LM -c 0.15")
  ## read results
  f = open(queryresultfile, 'r')
  andresults = []
  orresults = []
  for line in f:
    line = line.strip()
    parts = line.split()
    if len(parts) < 6:
      continue
    document = parts[2]
    score = float(parts[4])
    myQueryId = int(parts[0])
    if myQueryId <= queryIdAnd:
      andresults.append([document, score])
    else:
      orresults.append(document)
  f.close()
  os.chdir(pwd)

  andresults = sorted(andresults, key=lambda x:x[1], reverse=True)
 
  r = 0
  count = 0
  while count < docNo:
    if r < len(andresults):
      if not andresults[r][0] in resultingDocs:
        resultingDocs.append(andresults[r][0])
        count += 1
    else:
      rNew = r - len(andresults)
      if rNew < len(orresults):
        if not orresults[rNew] in resultingDocs:
          resultingDocs.append(orresults[rNew])
          count += 1
      else:
        break
    r += 1

  logger.info("IR result: " + str(len(resultingDocs)) + " documents")
  return resultingDocs 
