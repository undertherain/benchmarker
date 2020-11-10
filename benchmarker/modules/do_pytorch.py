import argparse
from timeit import default_timer as timer

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import mkldnn as mkldnn_utils
from torch.cuda import amp
from .torchprof import Profile

from .i_neural_net import INeuralNet


def progress(epoch, idx, nb, loss, log_interval=10):
    if idx % log_interval == 0:
        prc = 100.0 * idx / nb
        stat = f"{epoch} [{idx}/{nb} ({prc:.0f}%)]\tLoss: {loss:.6f}"
        print("Train Epoch: " + stat)


class Benchmark(INeuralNet):
    def __init__(self, params, extra_args=None):
        parser = argparse.ArgumentParser(description="pytorch extra args")
        parser.add_argument("--backend", default="native")
        parser.add_argument("--cudnn_benchmark", dest="cbm", action="store_true")
        parser.add_argument("--no_cudnn_benchmark", dest="cbm", action="store_false")
        parser.add_argument("--precision", default="FP32")
        parser.add_argument("--profile_pytorch", dest="profile", action="store_true")
        parser.set_defaults(cbm=True)
        args, remaining_args = parser.parse_known_args(extra_args)
        super().__init__(params, remaining_args)
        self.params["profile_pytorch"] = args.profile
        self.params["channels_first"] = True
        params["problem"]["precision"] = args.precision
        self.params["backend"] = args.backend
        self.params["cudnn_benchmark"] = args.cbm
        if self.params["nb_gpus"] > 0:
            if self.params["backend"] != "native":
                raise RuntimeError("only native backend is supported for GPUs")
            assert self.params["problem"]["precision"] in {"FP32", "FP16", "mixed"}
        else:
            assert self.params["problem"]["precision"] == "FP32"
        torch.backends.cudnn.benchmark = self.params["cudnn_benchmark"]
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
        else:
            if self.params["backend"] == "native":
                torch.backends.mkldnn.enabled = False
            else:
                raise RuntimeError("Unknown backend")
        x_train, y_train = self.load_data()
        self.device = torch.device("cuda" if self.params["gpus"] else "cpu")
        self.x_train = torch.from_numpy(x_train).to(self.device)
        self.y_train = torch.from_numpy(y_train).to(self.device)

    def train(self, model, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(zip(self.x_train, self.y_train)):
            optimizer.zero_grad()
            if self.params["problem"]["precision"] == "mixed":
                with amp.autocast():
                    loss = model(data, target)
            else:
                loss = model(data, target)

            loss.mean().backward()
            optimizer.step()
            progress(epoch, batch_idx, len(self.x_train), loss.mean().item())
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def set_random_seed(self, seed):
        super().set_random_seed(seed)
        torch.manual_seed(seed)

    def inference(self, model, device):
        # test_loss = 0
        # correct = 0
        with torch.no_grad():
            for data, target in zip(self.x_train, self.y_train):
                if self.params["backend"] == "DNNL":
                    data = data.to_mkldnn()
                # TODO: add option to keep data on GPU
                # data, target = data.to(device), target.to(device)
                output = model(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                # TODO: get back softmax for ResNet-like models
                # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

                # Profile using torchprof (TODO:profile_per_batch for all batches and epochs)
                if self.params["profile_pytorch"]:
                    profile_cuda = self.device.type == "cuda"
                    with Profile(model, use_cuda=profile_cuda) as prof:
                        model(data)
                    profile_output_as_dict = prof.display(show_events=False)
                    self.params["profile_data"] = profile_output_as_dict

        if self.params["nb_gpus"] > 0:
            torch.cuda.synchronize()

    def run_internal(self):
        model = self.net
        if len(self.params["gpus"]) > 1:
            model = nn.DataParallel(model)
        # TODO: make of/on-core optional

        model.to(self.device)
        # TODO: args for training hyperparameters
        start = timer()
        if self.params["problem"]["precision"] == "FP16":
            if self.x_train.dtype == torch.float32:
                self.x_train = self.x_train.half()
            if self.y_train.dtype == torch.float32:
                self.y_train = self.y_train.half()
            model.half()
        if self.params["mode"] == "training":
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
            if self.params["problem"]["precision"] == "mixed":
                assert len(self.params["gpus"]) == 1
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.train(model, optimizer, epoch)
        else:
            assert self.params["problem"]["precision"] in ["FP16", "FP32"]
            model.eval()
            if self.params["backend"] == "DNNL":
                model = mkldnn_utils.to_mkldnn(model)
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.inference(model, self.device)
        end = timer()
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        self.params["framework_full"] = "PyTorch-" + torch.__version__
        return self.params
