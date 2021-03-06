#!/bin/bash

allennlp train -f --include-package srl -s models/agent configs/agent.jsonnet
allennlp train -f --include-package srl -s models/experiencer configs/experiencer.jsonnet
allennlp train -f --include-package srl -s models/patient configs/patient.jsonnet
allennlp train -f --include-package srl -s models/recipient configs/recipient.jsonnet
allennlp train -f --include-package srl -s models/theme configs/theme.jsonnet
