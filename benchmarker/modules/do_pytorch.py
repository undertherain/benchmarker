import argparse
import logging
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import mkldnn as mkldnn_utils
from torch.cuda import amp
from .torchprof import Profile
from .i_neural_net import INeuralNet


logger = logging.getLogger(__name__)


def progress(epoch, idx, nb, loss, log_interval=10):
    if idx % log_interval == 0:
        prc = 100.0 * idx / nb
        stat = f"{epoch} [{idx}/{nb} ({prc:.0f}%)]\tLoss: {loss:.6f}"
        print("Train Epoch: " + stat)


class Benchmark(INeuralNet):
    def __init__(self, params, extra_args=None):
        args, remaining_args = self.parse_args(extra_args)
        super().__init__(params, remaining_args)
        self.params["profile_pytorch"] = args.profile
        self.params["channels_first"] = True
        params["problem"]["precision"] = args.precision
        self.params["backend"] = args.backend
        self.params["tensor_layout"] = args.tensor_layout
        self.params["cudnn_benchmark"] = args.cbm
        if self.params["nb_gpus"] > 0:
            if self.params["backend"] != "native":
                raise RuntimeError("only native backend is supported for GPUs")
            assert self.params["problem"]["precision"] in {"FP32", "FP16", "mixed"}
        else:
            assert self.params["problem"]["precision"] == "FP32"
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
        parser.add_argument("--precision", default="FP32")
        parser.add_argument("--profile_pytorch", dest="profile", action="store_true")
        parser.set_defaults(cbm=True)
        args, remaining_args = parser.parse_known_args(extra_args)
        return args, remaining_args

    def setup_data_and_model(self):
        x_train, y_train = self.load_data()
        self.x_train = [torch.from_numpy(x).to(self.device) for x in x_train]
        self.y_train = [torch.from_numpy(y).to(self.device) for y in y_train]
        if self.params["problem"]["precision"] == "FP16":
            self.net.half()
            if self.x_train[0].dtype == torch.float32:
                self.x_train = [x.half() for x in self.x_train]
            if self.y_train.dtype == torch.float32:
                self.y_train = [y.half() for y in self.y_train]
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
            self.net.eval()  # This is to make it not fail when DNLL does not support train
            if self.params["tensor_layout"] == "DNNL":
                self.net = mkldnn_utils.to_mkldnn(self.net)
                if self.x_train[0].dtype in [torch.float32, torch.float16]:
                    self.x_train = [x.to_mkldnn() for x in self.x_train]
                if self.y_train[0].dtype in [torch.float32, torch.float16]:
                    self.y_train = [y.to_mkldnn() for y in self.y_train]
                # TODO: check if softmax etc now works with DNNL
            else:
                logger.warning("Using DNNL backend without DNNL tensors")
        else:
            if self.params["backend"] == "native":
                torch.backends.mkldnn.enabled = False
                assert self.params["tensor_layout"] == "native"
            else:
                raise RuntimeError("Unknown backend")

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
        with torch.no_grad():
            # for data, target in zip(self.x_train, self.y_train):
            for i in range(len(self.x_train)):
                data = self.x_train[i]
                _ = model(data)
                # Profile using torchprof (TODO:profile_per_batch for all batches and epochs)
                if self.params["profile_pytorch"]:
                    profile_cuda = self.device.type == "cuda"
                    with Profile(model, use_cuda=profile_cuda) as prof:
                        model(data)
                    profile_output_as_dict = prof.display(show_events=False)
                    self.params["profile_data"] = profile_output_as_dict

        if self.params["nb_gpus"] > 0:
            torch.cuda.synchronize()

    def run(self):
        model = self.net
        if len(self.params["gpus"]) > 1:
            model = nn.DataParallel(model)

        model.to(self.device)
        # TODO: args for training hyperparameters
        start = timer()
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
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.inference(model, self.device)
        end = timer()
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        self.params["framework_full"] = "PyTorch-" + torch.__version__
        return self.params
