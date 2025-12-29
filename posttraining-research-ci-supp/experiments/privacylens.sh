#!/bin/bash

set -e

az ml job create -f privacylens.yaml \
  --set inputs.model.path=../models/mai-ds-r1 \
  --set inputs.engine=AzureOpenAI \
  --set inputs.cot=No  \
  --set settings.default_compute=azureml:E4sv3-EUS2-hipri \
  --set display_name=privacylens-mai_ds_r1

az ml job create -f privacylens.yaml \
  --set inputs.model.path=../models/mai-ds-r1 \
  --set inputs.engine=AzureOpenAI \
  --set inputs.cot=Yes  \
  --set settings.default_compute=azureml:E4sv3-EUS2-hipri \
  --set display_name=privacylens-mai_ds_r1

az ml job create -f privacylens.yaml \
  --set inputs.model.path=../models/phi4-reasoning \
  --set inputs.engine=AzureOpenAI \
  --set inputs.cot=No  \
  --set settings.default_compute=azureml:E4sv3-EUS2-hipri \
  --set display_name=privacylens-phi4_reasoning

az ml job create -f privacylens.yaml \
  --set inputs.model.path=../models/phi4-reasoning \
  --set inputs.engine=AzureOpenAI \
  --set inputs.cot=Yes  \
  --set settings.default_compute=azureml:E4sv3-EUS2-hipri \
  --set display_name=privacylens-phi4_reasoning-cot

az ml job create -f privacylens.yaml \
  --set inputs.model.path=azureml:Qwen-Qwen2_5-7B-Instruct:1 \
  --set inputs.engine=VLLM \
  --set inputs.cot=No \
  --set display_name=privacylens-qwen2_5-7b-instruct

az ml job create -f privacylens.yaml \
  --set inputs.model.path=azureml:Qwen-Qwen2_5-7B-Instruct:1 \
  --set inputs.engine=VLLM \
  --set inputs.cot=Yes \
  --set display_name=privacylens-qwen2_5-7b-instruct-cot
