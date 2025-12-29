# Reasoning For Contextual Integrity

This project is a research prototype for a contextual integrity (CI) based privacy framework for machine learning (ML) systems.

## Setup

These experiments are run on Azure ML.
Follow the [official Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public#installation) to set up your environment.

## Privacylens

See `experiments/privacylens.yaml` for the configuration of the privacylens experiment.

The option `cot` enables the chain-of-thought (CoT) prompt for the privacy lens experiment.

```bash
az ml job create -f experiments/privacylens.yaml --web
```

## RL Training

See `components/training/README.md` for the configuration of the RL training experiment.
