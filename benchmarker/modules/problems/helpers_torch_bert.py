import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class BertTraining(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, t):
        loss, _logits = self.net(input_ids=x,
                                 labels=t)
        return loss


class BertInference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x):
        logits = self.net(input_ids=x)
        return logits


def get_kernel_by_name(params, unparsed_args, name_model):
    assert unparsed_args == []
    config = AutoConfig.from_pretrained(name_model, num_labels=3)
    net = AutoModelForSequenceClassification.from_pretrained(name_model, config=config)
    if params["mode"] == "inference":
        return BertInference(net)
    else:
        return BertTraining(net)
    return net
