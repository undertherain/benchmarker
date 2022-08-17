import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class BertTraining(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, input_ids, labels):
        ret = self.net(input_ids=input_ids,
                       labels=labels)
        return ret["loss"]


class BertInference(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, input_ids, labels):
        logits = self.net(input_ids=input_ids)
        return logits


def get_kernel_by_name(params, name_model):
    config = AutoConfig.from_pretrained(name_model, num_labels=3)
    net = AutoModelForSequenceClassification.from_pretrained(name_model, config=config)
    if params["mode"] == "inference":
        return BertInference(net)
    else:
        return BertTraining(net)
    return net
