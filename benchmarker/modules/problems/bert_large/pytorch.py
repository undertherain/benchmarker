from transformers import AutoModelForSequenceClassification, AutoConfig
from benchmarker.modules.problems.helpers_torch_bert import BertInference, BertTraining


config = AutoConfig.from_pretrained(
    "bert-large-uncased",
    num_labels=3)


def get_kernel(params, unparsed_args=None):
    assert unparsed_args == []
    net = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", config=config)
    if params["mode"] == "inference":
        return BertInference(net)
    else:
        return BertTraining(net)
    return net
