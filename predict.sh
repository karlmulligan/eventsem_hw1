#!/bin/bash

export OUTPUT_FILE=predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         logs/test4 \
                           data/agent/test.json
