import argparse
import contextlib
import json
import logging
from timeit import default_timer as timer

import benchmarker.frameworks.patch_torch as patch_torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils import mkldnn as mkldnn_utils

from .i_neural_net import INeuralNet

logger = logging.getLogger(__name__)
from enum import Enum


class MyEnum(Enum):
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None

    def __str__(self):
        return self.value

class Numerics(MyEnum):
    fp16 = 'fp16'
    fp32 = 'fp32'
    fp64 = 'fp64'
    bf16 = 'bf16'

class Precision(MyEnum):
    medium = 'medium'
    high = 'high'
    highest = 'highest'


def progress(epoch, idx, nb, loss, log_interval=10):
    if idx % log_interval == 0:
        prc = 100.0 * idx / nb
        stat = f"{epoch} [{idx}/{nb} ({prc:.0f}%)]\tLoss: {loss:.6f}"
        print("Train Epoch: " + stat)


def set_tensor_device_precision(tensor, device, layout, numerics):
    # if isinstance(tensor, dict):
    #     return {k: set_tensor_device_precision(v) for k, v in tensor.items()}
    if isinstance(tensor, (np.ndarray, np.generic)):
        tensor = torch.from_numpy(tensor)
    if tensor.dtype == torch.float32:
        if numerics == "fp16":
            tensor = tensor.half()
        if numerics == "bf16":
            tensor = tensor.bfloat16()
    tensor = tensor.to(device)
    if tensor.dtype in [torch.float32, torch.float16]:
        if layout == "DNNL":
            tensor = tensor.to_mkldnn()
    return tensor


def set_batch_device_precision(data, device, layout, numerics):
    if isinstance(data, list):
        return [set_batch_device_precision(i, device, layout, numerics) for i in data]
    if isinstance(data, tuple):
        return (set_batch_device_precision(i, device, layout, numerics) for i in data)
    if isinstance(data, dict):
        for key, value in data.items():
            return {k: set_tensor_device_precision(v, device, layout, numerics) for k, v in data.items()}
    else:
        batch = set_tensor_device_precision(data, device, layout, numerics)
    return batch


class WrapperOneInput(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(**x)


class Benchmark(INeuralNet):
    def __init__(self, params, extra_args=None):
        args, remaining_args = self.parse_args(extra_args)
        super().__init__(params, remaining_args)
        self.params["profile_pytorch"] = args.profile
        self.params["channels_first"] = True
        self.params["problem"]["numerics"] = args.numerics.value
        self.params["problem"]["precision"] = args.precision.value
        self.params["backend"] = args.backend
        self.params["tensor_layout"] = args.tensor_layout
        self.params["cudnn_benchmark"] = args.cbm
        self.params["compile"] = args.compile
        if self.params["profile_pytorch"]:
            assert (
                self.params["mode"] == "inference"
            ), "--profile_pytorch works only with --mode=inference"
        if self.params["nb_gpus"] > 0:
            if self.params["backend"] != "native":
                raise RuntimeError("only native backend is supported for GPUs")
        # assert self.params["problem"]["numerics"] in {"FP16", "FP32", "FP64"}
        torch.backends.cudnn.benchmark = self.params["cudnn_benchmark"]
        self.device = torch.device("cuda" if self.params["gpus"] else "cpu")
        # TODO: make of/on-core optional
        self.setup_data_and_model()

    def parse_args(self, extra_args):
        parser = argparse.ArgumentParser(description="pytorch extra args")
        parser.add_argument("--backend", default="native")
        parser.add_argument("--tensor_layout", default="native")
        parser.add_argument("--cudnn_benchmark", dest="cbm", action="store_true")
        parser.add_argument("--no_cudnn_benchmark", dest="cbm", action="store_false")
        parser.add_argument("--numerics", type=Numerics, choices=list(Numerics))
        parser.add_argument("--precision", type=Precision, choices=list(Precision))
        parser.add_argument("--profile_pytorch", dest="profile", action="store_true")
        parser.add_argument("--compile", dest="compile", action="store_true")
        parser.set_defaults(cbm=True)
        args, remaining_args = parser.parse_known_args(extra_args)
        return args, remaining_args

    def setup_data_and_model(self):
        batches = self.load_data()
        # print("loaded", batches)
        args = [
            self.device,
            self.params["tensor_layout"],
            self.params["problem"]["numerics"],
        ]
        self.batches = set_batch_device_precision(batches, *args)
        # self.y_train = [set_batch_device_precision(i, *args) for i in y_train]
        if self.params["problem"]["numerics"] == "fp32":
            torch.set_float32_matmul_precision(self.params["problem"]["precision"])
            # torch.backends.cuda.matmul.allow_tf32 = True
            # torch.backends.cudnn.allow_tf32 = True
        #else:
            #torch.backends.cuda.matmul.allow_tf32 = False
            #torch.backends.cudnn.allow_tf32 = False
        elif self.params["problem"]["numerics"] == "fp16":
            self.net.half()
        elif self.params["problem"]["numerics"] == "bf16":
            self.net.bfloat16()
        else:
            raise ValueError("Unknown numerics " +  self.params["problem"]["numerics"])
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
            self.net.eval()  # This is to make it not fail when DNLL does not support train
            if self.params["tensor_layout"] == "DNNL":
                self.net = mkldnn_utils.to_mkldnn(self.net)
            else:
                logger.warning("Using DNNL backend without DNNL tensors")
        else:
            if self.params["backend"] == "native":
                torch.backends.mkldnn.enabled = False
                assert self.params["tensor_layout"] == "native"
            else:
                raise RuntimeError("Unknown backend")

    def train(self, model, optimizer, epoch):
        with amp.autocast() if self.params["problem"]["numerics"] == "mixed" else contextlib.suppress():
            model.train()
            for batch_idx, batch in enumerate(self.batches):
                optimizer.zero_grad()
                # print("batch")
                # print(self.batches)
                # print(batch.shape)
                # print(batch)
                # return
                loss = model(** batch)
                loss.backward()
                # loss.mean().backward()
                optimizer.step()
            progress(epoch, batch_idx, len(self.batches), loss.item())
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def set_random_seed(self, seed):
        super().set_random_seed(seed)
        torch.manual_seed(seed)

    def inference(self, model, device):
        with torch.no_grad():
            # for data, target in zip(self.x_train, self.y_train):
            if self.params["profile_pytorch"]:
                try:
                    from torch.profiler import profile

                    with profile(record_shapes=True) as prof:
                        self.inner_loop(model)
                    profile_data = prof.key_averages().table(
                        sort_by="cpu_time_total",
                        row_limit=20,
                    )
                    print(profile_data)
                except Exception:
                    from .torchprof import Profile

                    # Profile using torchprof (TODO:profile_per_batch for all batches and epochs)
                    profile_cuda = self.device.type == "cuda"
                    with Profile(model, use_cuda=profile_cuda) as prof:
                        self.inner_loop(model)
                    data = prof.display(show_events=False)
                    profile_data = json.dumps(data, indent=4, separators=(",", ": "))

                self.params["profile_data"] = profile_data
            else:
                self.inner_loop(model)

        if self.params["nb_gpus"] > 0:
            torch.cuda.synchronize()

    def inner_loop(self, model):
        for batch in self.batches:
            # print(batch["x"].dtype)
            # print(batch["labels"].dtype)
            # return
            _ = model(**batch)

    def get_batch_inference_flops(self):
        from fvcore.nn import FlopCountAnalysis
        model = WrapperOneInpput(self.net)
        flops = FlopCountAnalysis(model, self.batches[0])
        return flops.total()

    def run(self):
        # TODO: make it an option
        # patch_torch.patch_bmm()
        if self.params["compile"]:
            model = torch.compile(self.net)
            print("compiling done")
        else:
            model = self.net

        if len(self.params["gpus"]) > 1:
            model = nn.DataParallel(model)

        model.to(self.device)
        # TODO: args for training hyperparameters
        if self.params["mode"] == "training":
            model.train()
            # TODO: log optimizer to metadata / set from params
            # optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.95)
            optimizer = optim.AdamW(model.parameters(), lr=0.0001)
            if self.params["problem"]["numerics"] == "mixed":
                assert len(self.params["gpus"]) == 1
            if self.params["preheat"]:
                self.train(model, optimizer, 1)
        else:
            # assert self.params["problem"]["precision"] in ["FP16", "FP32"]
            model.eval()
            if self.params["preheat"]:
                self.inference(model, self.device)

        start = timer()
        for epoch in range(1, self.params["nb_epoch"] + 1):
            if self.params["mode"] == "training":
                self.train(model, optimizer, epoch)
            else:
                self.inference(model, self.device)
        end = timer()

        # TODO: make this a paramter
        # if self.params["flops"]:
        #    flops = self.get_batch_inference_flops()
        #    self.params["problem"]["gflop_estimated"] = flops * self.params["nb_epoch"] * self.params["problem"]["cnt_batches_per_epoch"] / 1000**3
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        self.params["framework_full"] = "PyTorch-" + torch.__version__
        return self.params
