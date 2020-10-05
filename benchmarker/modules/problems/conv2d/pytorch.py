import torch.nn as nn
import math
# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params):
    assert params["mode"] == "inference"
    print(params["problem"])
    problem_params = params["problem"]

    cnt_samples = problem_params["size"][0]
    num_channels = problem_params["size"][1]
    cnt_filters = problem_params["cnt_filters"]
    filter_width = filter_height = problem_params["size_kernel"]
    input_width = problem_params["size"][2]
    input_height = problem_params["size"][3]
    padding_width = padding_height = problem_params["padding"]
    horizontal_stride = vertical_stride = problem_params["stride"]
    out_width = 1 + math.floor((input_width - filter_width + 2 * padding_width)/horizontal_stride) 
    out_height = 1 + math.floor((input_height - filter_height + 2 * padding_height)/vertical_stride)
    params["problem"]["flop_estimated"] = 2 * out_width * out_height * num_channels * cnt_samples * cnt_filters * filter_width * filter_height * params["nb_epoch"]

    Net = nn.Conv2d(in_channels=num_channels,
                    out_channels=cnt_filters,
                    kernel_size=filter_width,
                    stride=horizontal_stride,
                    padding=padding_width,
                    dilation=problem_params["dilation"],
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return Net
