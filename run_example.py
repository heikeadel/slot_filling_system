#!/usr/bin/python
import time
import sys
from eval_CS2015 import Evaluation
queryfile = "queries_example" # replace with TAC query file
outputfile = "output.example" # this is where the results will be written to
thresholdfile="data/weightsAndThresholds/slot2threshold.PAT+skipSVM+CNN.binary"
weightfile="data/weightsAndThresholds/slot2weight.PAT+skipSVM+CNN.binary"
terrierDir = "terrier-dir" # replace with path to Terrier directory
run = Evaluation(queryfile, outputfile, terrierDir, doEntityLinking = 1, doCoref = 1, weightfile = weightfile, thresholdfile = thresholdfile, svmVersion="binarySkip", cnnVersion="binary") # doEntityLinking/doCoref: 0 (not use it) or 1 (use it), svmVersion: {binarySkip, binaryBOW, multiSkip}, cnnVersion: {binary, multi, pipeline, joint, global}
