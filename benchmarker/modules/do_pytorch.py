import argparse
from timeit import default_timer as timer

import torch
# TODO: should we expect an import error here?
# https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python
import torch.backends.mkldnn
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils import mkldnn as mkldnn_utils

# from torchvision import datasets, transforms
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
        args, remaining_args = parser.parse_known_args(extra_args)
        super().__init__(params, remaining_args)
        self.params["backend"] = args.backend
        if self.params["nb_gpus"] > 0:
            if self.params["backend"] != "native":
                raise RuntimeError("only native backend is supported for GPUs")

        self.params["channels_first"] = True

    def train(self, model, device, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(zip(self.x_train, self.y_train)):
            optimizer.zero_grad()
            loss = model(data, target)
            loss.mean().backward()
            optimizer.step()
            progress(epoch, batch_idx, len(self.x_train), loss.mean().item())
        if device.type == "cuda":
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
                # print(data.shape)
                output = model(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                # TODO: get back softmax for ResNet-like models
                # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()
        if self.params["nb_gpus"] > 0:
            torch.cuda.synchronize()
        # test_loss /= len(test_loader.dataset)

        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, len(test_loader.dataset),
        #    100. * correct / len(test_loader.dataset)))

    def run_internal(self):
        # use_cuda = not args.no_cuda and torch.cuda.is_available()
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
        else:
            if self.params["backend"] == "native":
                torch.backends.mkldnn.enabled = False
            else:
                raise RuntimeError("Unknown backend")
        device = torch.device("cuda" if self.params["gpus"] else "cpu")

        x_train, y_train = self.load_data()

        # train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=False)

        model = self.net
        if len(self.params["gpus"]) > 1:
            model = nn.DataParallel(model)
        # TODO: make of/on-core optional
        self.x_train = torch.from_numpy(x_train).to(device)
        self.y_train = torch.from_numpy(y_train).to(device)
        model.to(device)
        # TODO: args for training hyperparameters
        start = timer()
        if self.params["mode"] == "training":
            model.train()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.train(model, device, optimizer, epoch)
            # test(args, model, device, test_loader)
        else:
            model.eval()
            if self.params["backend"] == "DNNL":
                model = mkldnn_utils.to_mkldnn(model)
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.inference(model, device)
        end = timer()
        self.params["time"] = (end - start) / self.params["nb_epoch"]
        self.params["framework_full"] = "PyTorch-" + torch.__version__
        return self.params
