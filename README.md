# CIS Slot Filling System 2015

This is the code for the slot filling system of the CIS entry to the TAC KBP Slot Filling shared task in 2015.
It has been developed from 2014-2015.
Several scripts for slot filler classification (multi-class models) are newer (from 2016-2017).

Author: Heike Adel, 2014-2017

---------------------------------------

Prerequisites:

In order to run the slot filling systems, the following software is required:
- Terrier (Version 4.0) [1] -> enter the path to Terrier in doQuery.py
- Stanford CoreNLP [2] (Version "stanford-corenlp-full-2014-01-04") -> enter the path to CoreNLP in doNERandCoref.py and utilities.py
- Scikit Learn (Version 0.17.1)
- Theano (Version 0.8.0.dev0.dev-e088e2a8d81a27c98b2e80b1a05227700996e42d)
- editdist (Version 0.3 from http://www.mindrot.org/files/py-editdist)

Preparation:
- Add the path of Terrier and Stanford CoreNLP to the slot filling system (see above)
- Prepare the documents for indexing with Terrier (some cleaning steps) by running preprocessing/normalizeTexts_forTerrier.py
- Index the preprocessed documents with Terrier (see Terrier commands)
- If you would like to use entity linking with WAT [3], you need to obtain an authorization token at https://sobigdata.d4science.org/web/tagme/wat-api and enter it in modul_entityLinking.py
- adapt the following paths according to your file system:
  - doNerAndCoref.py and readNerAndCoref: TAGGING_PATH: this is where the results from CoreNLP will be stored
  - doQuery.py, run_example.py: TERRIER_DIR: this is the path to the indexed version of the evaluation corpus
  - data/docId2path_corpus2015: PATH_TO_CORPUS: this is the path to the 2015 evaluation corpus
  - eval_CS2015.py: PATH_TO_PATTERNS: this is the path to the patterns by Roth et al. 2013 [4]
---------------------------------------

Running the pipeline:
- The main class to call is Evaluation in eval_CS2015.py
- A sample running script is provided with run_example.py

---------------------------------------

Note:

- For using your own classification models inside the slot filling system, simply adapt modul_candEvaluation.py. 
- Training scripts for CNN and SVM can be found at preprocessing/trainingCNN and preprocessing/trainingSVM, respectively.
  - When using the config files in cnn/configs_X for training the CNNs, the paths for train and dev files need to be adapted
- Classification model weights and output thresholds can be tuned with preprocessing/getWeightsAndThresholds.py
- The CIS entry to SF 2015 used higher thresholds for the second hop results. The corresponding script is postprocessing/doHigherThresholds.py

---------------------------------------

Citation:

If you use this code or any part of it or any resources from this project, please cite:

@inproceedings{adelSF2015,
  author = {Heike Adel and Hinrich Sch\"{u}tze},
  title = {CIS at TAC Cold Start 2015: Neural Networks and Coreference Resolution for Slot Filling},
  booktitle = {Proceedings of Text Analysis Conference (TAC)},
  year = {2015}
}

If you use the binary CNN models, please cite:

@inproceedings{adelCNN2016,
  author = 	"Adel, Heike
		and Roth, Benjamin
		and Sch{\"u}tze, Hinrich",
  title = 	"Comparing Convolutional Neural Networks to Traditional Models for Slot Filling",
  booktitle = 	"Proceedings of the 2016 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies",
  year = 	"2016",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"828--838",
  location = 	"San Diego, California",
  doi = 	"10.18653/v1/N16-1097",
  url = 	"http://www.aclweb.org/anthology/N16-1097"
}

If you use the globally normalized CNN models, please cite:

@inproceedings{adelGlobal2017,
  author = 	"Adel, Heike
		and Sch{\"u}tze, Hinrich",
  title = 	"Global Normalization of Convolutional Neural Networks for Joint Entity and Relation Classification",
  booktitle = 	"Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1724--1730",
  location = 	"Copenhagen, Denmark",
  url = 	"http://www.aclweb.org/anthology/D17-1181"
}

---------------------------------------

References:
[1] Iadh Ounis, Gianni Amati, Vassilis Plachouras, Ben He, Craig Macdonald and Christina Lioma: "Terrier: A High Performance and Scalable Information Retrieval Platform", SIGIR Workshop on Open Source Information Retrieval (OSIR) 2006.
[2] Christopher D. Manning, Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky: "The Stanford CoreNLP Natural Language Processing Toolkit", ACL System Demonstrations 2014.
[3] Francesco Piccinno and Paolo Ferragina: "From Tagme to WAT: A New Entity Annotator", First International Workshop on Entity Recognition and Disambiguation 2014.
[4] Benjamin Roth, Tassilo Barth, Grzegorz Chrupała, Martin Gropp and Dietrich Klakow: "RelationFactory: A Fast, Modular and Effective System for Knowledge Base Population", EACL System Demonstrations 2014.

---------------------------------------

Contact:
heike.adel@gmail.com
