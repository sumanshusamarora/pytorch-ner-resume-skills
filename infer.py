import torch.nn as nn
import torch
import mlflow.pytorch


models = mlflow.pytorch.load_model(
            'file:///home/sam/work/research/ner-domain-specific/mlruns/1/39317e41864845f69ef9aae046baa420/artifacts/models')
models = models.to('cpu')
for i, data in enumerate(dataloader_test):
    if i > 1:
        out = models(data['x_padded'], data['x_char_padded'], data['x_postag_padded'])
        break