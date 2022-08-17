import torch


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, labels):
        return self.net(x)


def get_kernel(params):
    URL = (
        "https://api.ngc.nvidia.com/"
        f"v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    )

    ssd_cpu = torch.hub.load(
        "nvidia/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False
    )

    ckpt = torch.hub.load_state_dict_from_url(
        URL, map_location=lambda storage, loc: storage
    )
    ssd_cpu.load_state_dict(ckpt["model"])
    return Wrapper(ssd_cpu)
