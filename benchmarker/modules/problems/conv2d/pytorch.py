import torch.nn as nn
# import torch.nn.functional as F

# TODO: move this to params
cnt_channels = 3
size_image = 224
cnt_filters = 64
size_kernel = 3


Net = nn.Conv2d(in_channels=cnt_channels,
                out_channels=cnt_filters,
                kernel_size=size_kernel,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros')


# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs
def get_kernel(unparsed_args=None):
    return Net
