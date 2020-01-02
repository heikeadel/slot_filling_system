#!/usr/bin/python

#####################################
### CIS SLOT FILLING SYSTEM      ####
### 2014-2015                    ####
### Author: Heike Adel           ####
#####################################

import sys
import gzip
import os
import os.path
import re

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

outDir = "../data/normalizedForIndex"
if not os.path.exists(outDir):
  logger.info("creating output directory")
  os.makedirs(outDir)

def normalize(line):
  string = ""
  if "<doc id" in line:
    docId = re.sub(r'^\<doc id\=\"', '', line)
    docId = re.sub(r'\"\>\s*$', '', docId)
    string = "<DOC>\n"
    string += "<DOCNO>" + docId + "</DOCNO>\n<TEXT>"
  elif "<DOCID" in line:
    docId = re.sub(r"^\<DOCID\> ", '', line)
    docId = re.sub(r"\s*\<\/DOCID\>\s*$", '', docId)
    string = "<DOCNO>" + docId + "</DOCNO>\n<TEXT>"
  elif "<DOC id" in line:
    docId = re.sub(r'^\<DOC id\=\"', '', line)
    docId = re.sub(r'\" type.*\>\s*$', '', docId)
    string =  "<DOC>\n"
    string += "<DOCNO>" + docId + "</DOCNO>\n<TEXT>"
  elif "<DOC docid" in line:
    docId = re.sub(r'^\<DOC docid\=\"', '', line)
    docId = re.sub(r'\"\>\s*$', '', docId)
    string =  "<DOC>\n"
    string += "<DOCNO>" + docId + "</DOCNO>\n<TEXT>"
  elif "<DOC>" in line or "<doc>" in line:
    string = "<DOC>\n"
  else:
    if "</doc>" in line or "</DOC>" in line:
      string = "</TEXT>\n</DOC>"
    elif "<" in line and ">" in line:
      line = re.sub(r'\<', ' ', line)
      line = re.sub(r'\>', ' ', line)
      string = line
    elif "<" in line:
      # delete tag
      line = re.sub(r'\<(\/)*', ' ', line)
      string = line
    elif ">" in line:
      line = re.sub(r'\>', ' ', line)
    else:
      string = line
  if "&" in string: 
    string = re.sub(r'\&', 'AND', string)
  if u'\uFFFF' in str.decode('utf8'):
    string = re.sub(ur'\uFFFF', '', string)
  string = re.sub(r'\*', ' * ', string)
  string = re.sub(r'\"', ' " ', string)
  string = string.strip()
  return string

if len(sys.argv) != 2:
  logger.error("please pass the file to be normalized as input parameter")
  exit()

f = sys.argv[1]
if os.path.isfile(outDir + "/" + f):
  logger.warning("output file " + outDir + "/" + f + " already exists!")
if re.search(r"\.gz", f):
  curF = gzip.open(f, 'r')
  outFile = re.sub(r"\.gz$", '', f)
else:
  curF = open(f, 'r')
  outFile = f
logger.info("reading " + f)
out = open(outDir + "/" + outFile, 'w')
nextId = 2
out.write("<xml>\n")
printedLines = 1
for l in curF:
  result = normalize(l)
  out.write(result + "\n")
  printedLines += 1
out.write("</xml>\n")
out.close()
curF.close()
