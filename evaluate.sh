#!/bin/bash

allennlp evaluate --include-package srl --output-file metrics_agent.txt models/agent data/agent/test.json
allennlp evaluate --include-package srl --output-file metrics_experiencer.txt models/experiencer data/experiencer/test.json
allennlp evaluate --include-package srl --output-file metrics_patient.txt models/patient data/patient/test.json
allennlp evaluate --include-package srl --output-file metrics_recipient.txt models/recipient data/recipient/test.json
allennlp evaluate --include-package srl --output-file metrics_theme.txt  models/theme data/theme/test.json

