import numpy as np
import torch
import torch.nn as nn
from timeit import default_timer as timer
# import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from .i_neural_net import INeuralNet
import argparse
# TODO: should we expect an import error here? 
import torch.backends.mkldnn


class Benchmark(INeuralNet):
    def __init__(self, params, extra_args=None):
        parser = argparse.ArgumentParser(description='pytorch extra args')
        parser.add_argument('--backend', default="native")
        args, remaining_args = parser.parse_known_args(extra_args)
        super().__init__(params, remaining_args)
        self.params["backend"] = args.backend
        if self.params["nb_gpus"] > 0:
            if self.params["backend"] != "native":
                raise RuntimeError("only native backend is supported for GPUs")

        self.params["channels_first"] = True
        self.params["nb_epoch"] = 6

    def train(self, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print (output.shape, output[0][:10])
            # exit(-1)
            # loss = F.nll_loss(output, target)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            log_interval = 10
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        torch.cuda.synchronize()

    def inference(self, model, device):
        # test_loss = 0
        # correct = 0
        with torch.no_grad():
            for data, target in zip(self.x_train, self.y_train):
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

        #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, len(test_loader.dataset),
        #    100. * correct / len(test_loader.dataset)))

    def run_internal(self):
        # use_cuda = not args.no_cuda and torch.cuda.is_available()
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
        else:
            torch.backends.mkldnn.enabled = False

        if self.params["nb_gpus"] > 1:
            raise NotADirectoryError("multyple GPUs not supported yet")
        if self.params["gpus"]:
            torch.cuda.set_device(self.params["gpus"][0])
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        x_train, y_train = self.load_data()
        # TODO: make of/on-core optional
        self.x_train = torch.from_numpy(x_train).to(device)
        self.y_train = torch.from_numpy(y_train.astype(np.int64)).to(device)
        # train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=False)

        model = self.net.to(device)
        # TODO: args for training hyperparameters
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)
        start = timer()
        if self.params["mode"] == "training":
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.train(model, device, optimizer, epoch)
            # test(args, model, device, test_loader)
        else:
            model.eval()
            for epoch in range(1, self.params["nb_epoch"] + 1):
                self.inference(model, device)
        # TODO: return stats
        end = timer()
        self.params["time"] = (end - start) / self.params["nb_epoch"]
        self.params["framework_full"] = "PyTorch-" + torch.__version__
        return self.params
