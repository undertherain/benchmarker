import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=3)


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


def get_kernel(params, unparsed_args=None):
    assert unparsed_args == []
    net = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    if params["mode"] == "inference":
        return BertInference(net)
    else:
        return BertTraining(net)
    return net
