# from benchmarker.kernels.helpers_torch_bert import get_kernel_by_name
from transformers import ViTForImageClassification


def get_kernel(params):
    model_name = "facebook/deit-base-patch16-224"
    kernel = ViTForImageClassification.from_pretrained(model_name)
    return kernel
