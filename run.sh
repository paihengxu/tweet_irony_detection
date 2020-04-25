#!/usr/bin/env bash
cd stanford-corenlp-full-2018-10-05 || exit
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 5000 -timeout 10000
cd ..
python -m baseline.classify
