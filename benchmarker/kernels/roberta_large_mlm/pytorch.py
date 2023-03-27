from benchmarker.kernels.helpers_torch_bert import get_kernel_by_name


def get_kernel(params):
    return get_kernel_by_name(params,
                              name_model="roberta-large",
                              flavour="mlm")
