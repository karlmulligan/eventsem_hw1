#!/bin/bash

export OUTPUT_FILE=minpair_a_predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         models/agent \
                           data/agent/minpair.json

export OUTPUT_FILE=minpair_p_predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         models/patient \
                           data/patient/minpair.json

export OUTPUT_FILE=minpair_e_predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         models/experiencer \
                           data/experiencer/minpair.json

export OUTPUT_FILE=minpair_r_predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         models/recipient \
                           data/recipient/minpair.json

export OUTPUT_FILE=minpair_t_predictions.json

allennlp predict \
               --output-file $OUTPUT_FILE \
                 --include-package srl \
                   --predictor srl_predictor \
                     --use-dataset-reader \
                       --silent \
                         models/theme \
                           data/theme/minpair.json
