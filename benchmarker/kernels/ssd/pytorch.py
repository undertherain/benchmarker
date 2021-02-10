import torch

precision = 'fp32'
Net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                     'nvidia_ssd', model_math=precision)
