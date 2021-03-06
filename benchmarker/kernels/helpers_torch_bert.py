import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class BertTraining(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, t):
        ret = self.net(input_ids=x,
                       labels=t)
        return ret["loss"]


class BertInference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x):
        logits = self.net(input_ids=x)
        return logits


def get_kernel_by_name(params, name_model):
    config = AutoConfig.from_pretrained(name_model, num_labels=3)
    net = AutoModelForSequenceClassification.from_pretrained(name_model, config=config)
    if params["mode"] == "inference":
        return BertInference(net)
    else:
        return BertTraining(net)
    return net
