from benchmarker.kernels.helpers_torch_bert import get_kernel_by_name


def get_kernel(params, unparsed_args=None):
    return get_kernel_by_name(params, "bert-large-uncased")
